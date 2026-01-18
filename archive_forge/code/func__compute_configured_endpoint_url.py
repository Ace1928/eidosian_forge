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
def _compute_configured_endpoint_url(self, client_config, endpoint_url):
    if endpoint_url is not None:
        return endpoint_url
    if self._ignore_configured_endpoint_urls(client_config):
        logger.debug('Ignoring configured endpoint URLs.')
        return endpoint_url
    return self._config_store.get_config_variable('endpoint_url')