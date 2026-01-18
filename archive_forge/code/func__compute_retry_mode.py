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
def _compute_retry_mode(self, config_kwargs):
    retries = config_kwargs.get('retries')
    if retries is None:
        retries = {}
        config_kwargs['retries'] = retries
    elif 'mode' in retries:
        return
    retry_mode = self._config_store.get_config_variable('retry_mode')
    if retry_mode is None:
        retry_mode = 'legacy'
    retries['mode'] = retry_mode