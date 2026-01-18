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
def _register_smart_defaults_factory(self):

    def create_smart_defaults_factory():
        default_config_resolver = self._get_internal_component('default_config_resolver')
        imds_region_provider = IMDSRegionProvider(session=self)
        return SmartDefaultsConfigStoreFactory(default_config_resolver, imds_region_provider)
    self._internal_components.lazy_register_component('smart_defaults_factory', create_smart_defaults_factory)