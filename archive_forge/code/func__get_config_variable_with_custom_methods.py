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
def _get_config_variable_with_custom_methods(self, logical_name, methods):
    chain_builder = SubsetChainConfigFactory(session=self, methods=methods)
    mapping = create_botocore_default_config_mapping(self)
    for name, config_options in self.session_var_map.items():
        config_name, env_vars, default, typecast = config_options
        build_chain_config_args = {'conversion_func': typecast, 'default': default}
        if 'instance' in methods:
            build_chain_config_args['instance_name'] = name
        if 'env' in methods:
            build_chain_config_args['env_var_names'] = env_vars
        if 'config' in methods:
            build_chain_config_args['config_property_name'] = config_name
        mapping[name] = chain_builder.create_config_chain(**build_chain_config_args)
    config_store_component = ConfigValueStore(mapping=mapping)
    value = config_store_component.get_config_variable(logical_name)
    return value