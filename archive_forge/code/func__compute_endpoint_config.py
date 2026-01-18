import copy
import logging
import socket
import botocore.exceptions
import botocore.parsers
import botocore.serialize
from botocore.config import Config
from botocore.endpoint import EndpointCreator
from botocore.regions import EndpointResolverBuiltins as EPRBuiltins
from botocore.regions import EndpointRulesetResolver
from botocore.signers import RequestSigner
from botocore.useragent import UserAgentString
from botocore.utils import ensure_boolean, is_s3_accelerate_url
def _compute_endpoint_config(self, service_name, region_name, endpoint_url, is_secure, endpoint_bridge, s3_config):
    resolve_endpoint_kwargs = {'service_name': service_name, 'region_name': region_name, 'endpoint_url': endpoint_url, 'is_secure': is_secure, 'endpoint_bridge': endpoint_bridge}
    if service_name == 's3':
        return self._compute_s3_endpoint_config(s3_config=s3_config, **resolve_endpoint_kwargs)
    if service_name == 'sts':
        return self._compute_sts_endpoint_config(**resolve_endpoint_kwargs)
    return self._resolve_endpoint(**resolve_endpoint_kwargs)