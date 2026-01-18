from __future__ import (absolute_import, division, print_function)
import glob
import os
import os.path
import pkgutil
import sys
import warnings
from collections import defaultdict, namedtuple
from traceback import format_exc
import ansible.module_utils.compat.typing as t
from .filter import AnsibleJinja2Filter
from .test import AnsibleJinja2Test
from ansible import __version__ as ansible_version
from ansible import constants as C
from ansible.errors import AnsibleError, AnsiblePluginCircularRedirect, AnsiblePluginRemovedError, AnsibleCollectionUnsupportedVersionError
from ansible.module_utils.common.text.converters import to_bytes, to_text, to_native
from ansible.module_utils.compat.importlib import import_module
from ansible.module_utils.six import string_types
from ansible.parsing.utils.yaml import from_yaml
from ansible.parsing.yaml.loader import AnsibleLoader
from ansible.plugins import get_plugin_class, MODULE_CACHE, PATH_CACHE, PLUGIN_PATH_CACHE
from ansible.utils.collection_loader import AnsibleCollectionConfig, AnsibleCollectionRef
from ansible.utils.collection_loader._collection_finder import _AnsibleCollectionFinder, _get_collection_metadata
from ansible.utils.display import Display
from ansible.utils.plugin_docs import add_fragments
from ansible.utils.unsafe_proxy import _is_unsafe
import importlib.util
def _load_config_defs(self, name, module, path):
    """ Reads plugin docs to find configuration setting definitions, to push to config manager for later use """
    if self.class_name:
        type_name = get_plugin_class(self.class_name)
        if type_name in C.CONFIGURABLE_PLUGINS and (not C.config.has_configuration_definition(type_name, name)):
            dstring = AnsibleLoader(getattr(module, 'DOCUMENTATION', ''), file_name=path).get_single_data()
            if dstring:
                add_fragments(dstring, path, fragment_loader=fragment_loader, is_module=type_name == 'module')
                if 'options' in dstring and isinstance(dstring['options'], dict):
                    C.config.initialize_plugin_configuration_definitions(type_name, name, dstring['options'])
                    display.debug('Loaded config def from plugin (%s/%s)' % (type_name, name))