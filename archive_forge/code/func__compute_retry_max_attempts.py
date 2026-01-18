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
def _compute_retry_max_attempts(self, config_kwargs):
    retries = config_kwargs.get('retries')
    if retries is not None:
        if 'total_max_attempts' in retries:
            retries.pop('max_attempts', None)
            return
        if 'max_attempts' in retries:
            value = retries.pop('max_attempts')
            retries['total_max_attempts'] = value + 1
            return
    max_attempts = self._config_store.get_config_variable('max_attempts')
    if max_attempts is not None:
        if retries is None:
            retries = {}
            config_kwargs['retries'] = retries
        retries['total_max_attempts'] = max_attempts