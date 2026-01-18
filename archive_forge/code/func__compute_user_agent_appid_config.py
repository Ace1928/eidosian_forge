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
def _compute_user_agent_appid_config(self, config_kwargs):
    user_agent_appid = config_kwargs.get('user_agent_appid')
    if user_agent_appid is None:
        user_agent_appid = self._config_store.get_config_variable('user_agent_appid')
    if user_agent_appid is not None and len(user_agent_appid) > USERAGENT_APPID_MAXLEN:
        logger.warning(f'The configured value for user_agent_appid exceeds the maximum length of {USERAGENT_APPID_MAXLEN} characters.')
    config_kwargs['user_agent_appid'] = user_agent_appid