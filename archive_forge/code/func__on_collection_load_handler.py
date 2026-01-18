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
def _on_collection_load_handler(collection_name, collection_path):
    display.vvvv(to_text('Loading collection {0} from {1}'.format(collection_name, collection_path)))
    collection_meta = _get_collection_metadata(collection_name)
    try:
        if not _does_collection_support_ansible_version(collection_meta.get('requires_ansible', ''), ansible_version):
            mismatch_behavior = C.config.get_config_value('COLLECTIONS_ON_ANSIBLE_VERSION_MISMATCH')
            message = 'Collection {0} does not support Ansible version {1}'.format(collection_name, ansible_version)
            if mismatch_behavior == 'warning':
                display.warning(message)
            elif mismatch_behavior == 'error':
                raise AnsibleCollectionUnsupportedVersionError(message)
    except AnsibleError:
        raise
    except Exception as ex:
        display.warning('Error parsing collection metadata requires_ansible value from collection {0}: {1}'.format(collection_name, ex))