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
def _set_global_sts_endpoint(self, endpoint_config, is_secure):
    scheme = 'https' if is_secure else 'http'
    endpoint_config['endpoint_url'] = '%s://sts.amazonaws.com' % scheme
    endpoint_config['signing_region'] = 'us-east-1'