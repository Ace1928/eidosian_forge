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
def _query_collection_routing_meta(self, acr, plugin_type, extension=None):
    collection_pkg = import_module(acr.n_python_collection_package_name)
    if not collection_pkg:
        return None
    try:
        import_module(acr.n_python_collection_package_name + '.plugins.{0}'.format(plugin_type))
    except ImportError:
        pass
    collection_meta = getattr(collection_pkg, '_collection_meta', None)
    if not collection_meta:
        return None
    if acr.subdirs:
        subdir_qualified_resource = '.'.join([acr.subdirs, acr.resource])
    else:
        subdir_qualified_resource = acr.resource
    entry = collection_meta.get('plugin_routing', {}).get(plugin_type, {}).get(subdir_qualified_resource + extension, None)
    if not entry:
        entry = collection_meta.get('plugin_routing', {}).get(plugin_type, {}).get(subdir_qualified_resource, None)
    return entry