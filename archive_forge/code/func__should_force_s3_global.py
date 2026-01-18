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
def _should_force_s3_global(self, region_name, s3_config):
    s3_regional_config = 'legacy'
    if s3_config and 'us_east_1_regional_endpoint' in s3_config:
        s3_regional_config = s3_config['us_east_1_regional_endpoint']
        self._validate_s3_regional_config(s3_regional_config)
    is_global_region = region_name in ('us-east-1', None)
    return s3_regional_config == 'legacy' and is_global_region