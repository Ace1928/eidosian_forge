import base64
import binascii
import datetime
import email.message
import functools
import hashlib
import io
import logging
import os
import random
import re
import socket
import time
import warnings
import weakref
from datetime import datetime as _DatetimeClass
from ipaddress import ip_address
from pathlib import Path
from urllib.request import getproxies, proxy_bypass
import dateutil.parser
from dateutil.tz import tzutc
from urllib3.exceptions import LocationParseError
import botocore
import botocore.awsrequest
import botocore.httpsession
from botocore.compat import HEX_PAT  # noqa: F401
from botocore.compat import IPV4_PAT  # noqa: F401
from botocore.compat import IPV6_ADDRZ_PAT  # noqa: F401
from botocore.compat import IPV6_PAT  # noqa: F401
from botocore.compat import LS32_PAT  # noqa: F401
from botocore.compat import UNRESERVED_PAT  # noqa: F401
from botocore.compat import ZONE_ID_PAT  # noqa: F401
from botocore.compat import (
from botocore.exceptions import (
class S3EndpointSetter:
    _DEFAULT_PARTITION = 'aws'
    _DEFAULT_DNS_SUFFIX = 'amazonaws.com'

    def __init__(self, endpoint_resolver, region=None, s3_config=None, endpoint_url=None, partition=None, use_fips_endpoint=False):
        self._endpoint_resolver = endpoint_resolver
        self._region = region
        self._s3_config = s3_config
        self._use_fips_endpoint = use_fips_endpoint
        if s3_config is None:
            self._s3_config = {}
        self._endpoint_url = endpoint_url
        self._partition = partition
        if partition is None:
            self._partition = self._DEFAULT_PARTITION

    def register(self, event_emitter):
        event_emitter.register('before-sign.s3', self.set_endpoint)
        event_emitter.register('choose-signer.s3', self.set_signer)
        event_emitter.register('before-call.s3.WriteGetObjectResponse', self.update_endpoint_to_s3_object_lambda)

    def update_endpoint_to_s3_object_lambda(self, params, context, **kwargs):
        if self._use_accelerate_endpoint:
            raise UnsupportedS3ConfigurationError(msg='S3 client does not support accelerate endpoints for S3 Object Lambda operations')
        self._override_signing_name(context, 's3-object-lambda')
        if self._endpoint_url:
            return
        resolver = self._endpoint_resolver
        resolved = resolver.construct_endpoint('s3-object-lambda', self._region)
        new_endpoint = 'https://{host_prefix}{hostname}'.format(host_prefix=params['host_prefix'], hostname=resolved['hostname'])
        params['url'] = _get_new_endpoint(params['url'], new_endpoint, False)

    def set_endpoint(self, request, **kwargs):
        if self._use_accesspoint_endpoint(request):
            self._validate_accesspoint_supported(request)
            self._validate_fips_supported(request)
            self._validate_global_regions(request)
            region_name = self._resolve_region_for_accesspoint_endpoint(request)
            self._resolve_signing_name_for_accesspoint_endpoint(request)
            self._switch_to_accesspoint_endpoint(request, region_name)
            return
        if self._use_accelerate_endpoint:
            if self._use_fips_endpoint:
                raise UnsupportedS3ConfigurationError(msg='Client is configured to use the FIPS psuedo region for "%s", but S3 Accelerate does not have any FIPS compatible endpoints.' % self._region)
            switch_host_s3_accelerate(request=request, **kwargs)
        if self._s3_addressing_handler:
            self._s3_addressing_handler(request=request, **kwargs)

    def _use_accesspoint_endpoint(self, request):
        return 's3_accesspoint' in request.context

    def _validate_fips_supported(self, request):
        if not self._use_fips_endpoint:
            return
        if 'fips' in request.context['s3_accesspoint']['region']:
            raise UnsupportedS3AccesspointConfigurationError(msg={'Invalid ARN, FIPS region not allowed in ARN.'})
        if 'outpost_name' in request.context['s3_accesspoint']:
            raise UnsupportedS3AccesspointConfigurationError(msg='Client is configured to use the FIPS psuedo-region "%s", but outpost ARNs do not support FIPS endpoints.' % self._region)
        accesspoint_region = request.context['s3_accesspoint']['region']
        if accesspoint_region != self._region:
            if not self._s3_config.get('use_arn_region', True):
                raise UnsupportedS3AccesspointConfigurationError(msg='Client is configured to use the FIPS psuedo-region for "%s", but the access-point ARN provided is for the "%s" region. For clients using a FIPS psuedo-region calls to access-point ARNs in another region are not allowed.' % (self._region, accesspoint_region))

    def _validate_global_regions(self, request):
        if self._s3_config.get('use_arn_region', True):
            return
        if self._region in ['aws-global', 's3-external-1']:
            raise UnsupportedS3AccesspointConfigurationError(msg='Client is configured to use the global psuedo-region "%s". When providing access-point ARNs a regional endpoint must be specified.' % self._region)

    def _validate_accesspoint_supported(self, request):
        if self._use_accelerate_endpoint:
            raise UnsupportedS3AccesspointConfigurationError(msg='Client does not support s3 accelerate configuration when an access-point ARN is specified.')
        request_partition = request.context['s3_accesspoint']['partition']
        if request_partition != self._partition:
            raise UnsupportedS3AccesspointConfigurationError(msg='Client is configured for "%s" partition, but access-point ARN provided is for "%s" partition. The client and  access-point partition must be the same.' % (self._partition, request_partition))
        s3_service = request.context['s3_accesspoint'].get('service')
        if s3_service == 's3-object-lambda' and self._s3_config.get('use_dualstack_endpoint'):
            raise UnsupportedS3AccesspointConfigurationError(msg='Client does not support s3 dualstack configuration when an S3 Object Lambda access point ARN is specified.')
        outpost_name = request.context['s3_accesspoint'].get('outpost_name')
        if outpost_name and self._s3_config.get('use_dualstack_endpoint'):
            raise UnsupportedS3AccesspointConfigurationError(msg='Client does not support s3 dualstack configuration when an outpost ARN is specified.')
        self._validate_mrap_s3_config(request)

    def _validate_mrap_s3_config(self, request):
        if not is_global_accesspoint(request.context):
            return
        if self._s3_config.get('s3_disable_multiregion_access_points'):
            raise UnsupportedS3AccesspointConfigurationError(msg='Invalid configuration, Multi-Region Access Point ARNs are disabled.')
        elif self._s3_config.get('use_dualstack_endpoint'):
            raise UnsupportedS3AccesspointConfigurationError(msg='Client does not support s3 dualstack configuration when a Multi-Region Access Point ARN is specified.')

    def _resolve_region_for_accesspoint_endpoint(self, request):
        if is_global_accesspoint(request.context):
            self._override_signing_region(request, '*')
        elif self._s3_config.get('use_arn_region', True):
            accesspoint_region = request.context['s3_accesspoint']['region']
            self._override_signing_region(request, accesspoint_region)
            return accesspoint_region
        return self._region

    def set_signer(self, context, **kwargs):
        if is_global_accesspoint(context):
            if HAS_CRT:
                return 's3v4a'
            else:
                raise MissingDependencyException(msg='Using S3 with an MRAP arn requires an additional dependency. You will need to pip install botocore[crt] before proceeding.')

    def _resolve_signing_name_for_accesspoint_endpoint(self, request):
        accesspoint_service = request.context['s3_accesspoint']['service']
        self._override_signing_name(request.context, accesspoint_service)

    def _switch_to_accesspoint_endpoint(self, request, region_name):
        original_components = urlsplit(request.url)
        accesspoint_endpoint = urlunsplit((original_components.scheme, self._get_netloc(request.context, region_name), self._get_accesspoint_path(original_components.path, request.context), original_components.query, ''))
        logger.debug(f'Updating URI from {request.url} to {accesspoint_endpoint}')
        request.url = accesspoint_endpoint

    def _get_netloc(self, request_context, region_name):
        if is_global_accesspoint(request_context):
            return self._get_mrap_netloc(request_context)
        else:
            return self._get_accesspoint_netloc(request_context, region_name)

    def _get_mrap_netloc(self, request_context):
        s3_accesspoint = request_context['s3_accesspoint']
        region_name = 's3-global'
        mrap_netloc_components = [s3_accesspoint['name']]
        if self._endpoint_url:
            endpoint_url_netloc = urlsplit(self._endpoint_url).netloc
            mrap_netloc_components.append(endpoint_url_netloc)
        else:
            partition = s3_accesspoint['partition']
            mrap_netloc_components.extend(['accesspoint', region_name, self._get_partition_dns_suffix(partition)])
        return '.'.join(mrap_netloc_components)

    def _get_accesspoint_netloc(self, request_context, region_name):
        s3_accesspoint = request_context['s3_accesspoint']
        accesspoint_netloc_components = ['{}-{}'.format(s3_accesspoint['name'], s3_accesspoint['account'])]
        outpost_name = s3_accesspoint.get('outpost_name')
        if self._endpoint_url:
            if outpost_name:
                accesspoint_netloc_components.append(outpost_name)
            endpoint_url_netloc = urlsplit(self._endpoint_url).netloc
            accesspoint_netloc_components.append(endpoint_url_netloc)
        else:
            if outpost_name:
                outpost_host = [outpost_name, 's3-outposts']
                accesspoint_netloc_components.extend(outpost_host)
            elif s3_accesspoint['service'] == 's3-object-lambda':
                component = self._inject_fips_if_needed('s3-object-lambda', request_context)
                accesspoint_netloc_components.append(component)
            else:
                component = self._inject_fips_if_needed('s3-accesspoint', request_context)
                accesspoint_netloc_components.append(component)
            if self._s3_config.get('use_dualstack_endpoint'):
                accesspoint_netloc_components.append('dualstack')
            accesspoint_netloc_components.extend([region_name, self._get_dns_suffix(region_name)])
        return '.'.join(accesspoint_netloc_components)

    def _inject_fips_if_needed(self, component, request_context):
        if self._use_fips_endpoint:
            return '%s-fips' % component
        return component

    def _get_accesspoint_path(self, original_path, request_context):
        name = request_context['s3_accesspoint']['name']
        return original_path.replace('/' + name, '', 1) or '/'

    def _get_partition_dns_suffix(self, partition_name):
        dns_suffix = self._endpoint_resolver.get_partition_dns_suffix(partition_name)
        if dns_suffix is None:
            dns_suffix = self._DEFAULT_DNS_SUFFIX
        return dns_suffix

    def _get_dns_suffix(self, region_name):
        resolved = self._endpoint_resolver.construct_endpoint('s3', region_name)
        dns_suffix = self._DEFAULT_DNS_SUFFIX
        if resolved and 'dnsSuffix' in resolved:
            dns_suffix = resolved['dnsSuffix']
        return dns_suffix

    def _override_signing_region(self, request, region_name):
        signing_context = request.context.get('signing', {})
        signing_context['region'] = region_name
        request.context['signing'] = signing_context

    def _override_signing_name(self, context, signing_name):
        signing_context = context.get('signing', {})
        signing_context['signing_name'] = signing_name
        context['signing'] = signing_context

    @CachedProperty
    def _use_accelerate_endpoint(self):
        if self._s3_config.get('use_accelerate_endpoint'):
            return True
        if self._endpoint_url is None:
            return False
        netloc = urlsplit(self._endpoint_url).netloc
        if not netloc.endswith('amazonaws.com'):
            return False
        parts = netloc.split('.')
        if parts[0] != 's3-accelerate':
            return False
        feature_parts = parts[1:-2]
        if len(feature_parts) != len(set(feature_parts)):
            return False
        return all((p in S3_ACCELERATE_WHITELIST for p in feature_parts))

    @CachedProperty
    def _addressing_style(self):
        if self._use_accelerate_endpoint:
            return 'virtual'
        configured_addressing_style = self._s3_config.get('addressing_style')
        if configured_addressing_style:
            return configured_addressing_style

    @CachedProperty
    def _s3_addressing_handler(self):
        if self._addressing_style == 'virtual':
            logger.debug('Using S3 virtual host style addressing.')
            return switch_to_virtual_host_style
        if self._addressing_style == 'path' or self._endpoint_url is not None:
            logger.debug('Using S3 path style addressing.')
            return None
        logger.debug('Defaulting to S3 virtual host style addressing with path style addressing fallback.')
        return fix_s3_host