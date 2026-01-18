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
def _validate_min_compression_size(self, min_size):
    min_allowed_min_size = 1
    max_allowed_min_size = 1048576
    if min_size is not None:
        error_msg_base = f'Invalid value "{min_size}" for request_min_compression_size_bytes.'
        try:
            min_size = int(min_size)
        except (ValueError, TypeError):
            msg = f'{error_msg_base} Value must be an integer. Received {type(min_size)} instead.'
            raise botocore.exceptions.InvalidConfigError(error_msg=msg)
        if not min_allowed_min_size <= min_size <= max_allowed_min_size:
            msg = f'{error_msg_base} Value must be between {min_allowed_min_size} and {max_allowed_min_size}.'
            raise botocore.exceptions.InvalidConfigError(error_msg=msg)
    return min_size