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
class S3ArnParamHandler:
    _RESOURCE_REGEX = re.compile('^(?P<resource_type>accesspoint|outpost)[/:](?P<resource_name>.+)$')
    _OUTPOST_RESOURCE_REGEX = re.compile('^(?P<outpost_name>[a-zA-Z0-9\\-]{1,63})[/:]accesspoint[/:](?P<accesspoint_name>[a-zA-Z0-9\\-]{1,63}$)')
    _BLACKLISTED_OPERATIONS = ['CreateBucket']

    def __init__(self, arn_parser=None):
        self._arn_parser = arn_parser
        if arn_parser is None:
            self._arn_parser = ArnParser()

    def register(self, event_emitter):
        event_emitter.register('before-parameter-build.s3', self.handle_arn)

    def handle_arn(self, params, model, context, **kwargs):
        if model.name in self._BLACKLISTED_OPERATIONS:
            return
        arn_details = self._get_arn_details_from_bucket_param(params)
        if arn_details is None:
            return
        if arn_details['resource_type'] == 'accesspoint':
            self._store_accesspoint(params, context, arn_details)
        elif arn_details['resource_type'] == 'outpost':
            self._store_outpost(params, context, arn_details)

    def _get_arn_details_from_bucket_param(self, params):
        if 'Bucket' in params:
            try:
                arn = params['Bucket']
                arn_details = self._arn_parser.parse_arn(arn)
                self._add_resource_type_and_name(arn, arn_details)
                return arn_details
            except InvalidArnException:
                pass
        return None

    def _add_resource_type_and_name(self, arn, arn_details):
        match = self._RESOURCE_REGEX.match(arn_details['resource'])
        if match:
            arn_details['resource_type'] = match.group('resource_type')
            arn_details['resource_name'] = match.group('resource_name')
        else:
            raise UnsupportedS3ArnError(arn=arn)

    def _store_accesspoint(self, params, context, arn_details):
        params['Bucket'] = arn_details['resource_name']
        context['s3_accesspoint'] = {'name': arn_details['resource_name'], 'account': arn_details['account'], 'partition': arn_details['partition'], 'region': arn_details['region'], 'service': arn_details['service']}

    def _store_outpost(self, params, context, arn_details):
        resource_name = arn_details['resource_name']
        match = self._OUTPOST_RESOURCE_REGEX.match(resource_name)
        if not match:
            raise UnsupportedOutpostResourceError(resource_name=resource_name)
        accesspoint_name = match.group('accesspoint_name')
        params['Bucket'] = accesspoint_name
        context['s3_accesspoint'] = {'outpost_name': match.group('outpost_name'), 'name': accesspoint_name, 'account': arn_details['account'], 'partition': arn_details['partition'], 'region': arn_details['region'], 'service': arn_details['service']}