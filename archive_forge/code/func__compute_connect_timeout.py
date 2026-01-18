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
def _compute_connect_timeout(self, config_kwargs):
    connect_timeout = config_kwargs.get('connect_timeout')
    if connect_timeout is not None:
        return
    connect_timeout = self._config_store.get_config_variable('connect_timeout')
    if connect_timeout:
        config_kwargs['connect_timeout'] = connect_timeout