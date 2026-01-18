import copy
import logging
import os
import platform
import socket
import warnings
import botocore.client
import botocore.configloader
import botocore.credentials
import botocore.tokens
from botocore import (
from botocore.compat import HAS_CRT, MutableMapping
from botocore.configprovider import (
from botocore.errorfactory import ClientExceptionsFactory
from botocore.exceptions import (
from botocore.hooks import (
from botocore.loaders import create_loader
from botocore.model import ServiceModel
from botocore.parsers import ResponseParserFactory
from botocore.regions import EndpointResolver
from botocore.useragent import UserAgentString
from botocore.utils import (
from botocore.compat import HAS_CRT  # noqa
class SessionVarDict(MutableMapping):

    def __init__(self, session, session_vars):
        self._session = session
        self._store = copy.copy(session_vars)

    def __getitem__(self, key):
        return self._store[key]

    def __setitem__(self, key, value):
        self._store[key] = value
        self._update_config_store_from_session_vars(key, value)

    def __delitem__(self, key):
        del self._store[key]

    def __iter__(self):
        return iter(self._store)

    def __len__(self):
        return len(self._store)

    def _update_config_store_from_session_vars(self, logical_name, config_options):
        config_chain_builder = ConfigChainFactory(session=self._session)
        config_name, env_vars, default, typecast = config_options
        config_store = self._session.get_component('config_store')
        config_store.set_config_provider(logical_name, config_chain_builder.create_config_chain(instance_name=logical_name, env_var_names=env_vars, config_property_names=config_name, default=default, conversion_func=typecast))