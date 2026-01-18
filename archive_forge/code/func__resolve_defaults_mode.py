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
def _resolve_defaults_mode(self, client_config, config_store):
    mode = config_store.get_config_variable('defaults_mode')
    if client_config and client_config.defaults_mode:
        mode = client_config.defaults_mode
    default_config_resolver = self._get_internal_component('default_config_resolver')
    default_modes = default_config_resolver.get_default_modes()
    lmode = mode.lower()
    if lmode not in default_modes:
        raise InvalidDefaultsMode(mode=mode, valid_modes=', '.join(default_modes))
    return lmode