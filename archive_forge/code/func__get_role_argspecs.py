from __future__ import (absolute_import, division, print_function)
import os
from collections.abc import Container, Mapping, Set, Sequence
from types import MappingProxyType
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleParserError, AnsibleAssertionError
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six import binary_type, text_type
from ansible.playbook.attribute import FieldAttribute
from ansible.playbook.base import Base
from ansible.playbook.collectionsearch import CollectionSearch
from ansible.playbook.conditional import Conditional
from ansible.playbook.delegatable import Delegatable
from ansible.playbook.helpers import load_list_of_blocks
from ansible.playbook.role.metadata import RoleMetadata
from ansible.playbook.taggable import Taggable
from ansible.plugins.loader import add_all_plugin_dirs
from ansible.utils.collection_loader import AnsibleCollectionConfig
from ansible.utils.path import is_subpath
from ansible.utils.sentinel import Sentinel
from ansible.utils.vars import combine_vars
def _get_role_argspecs(self):
    """Get the role argument spec data.

        Role arg specs can be in one of two files in the role meta subdir: argument_specs.yml
        or main.yml. The former has precedence over the latter. Data is not combined
        between the files.

        :returns: A dict of all data under the top-level ``argument_specs`` YAML key
            in the argument spec file. An empty dict is returned if there is no
            argspec data.
        """
    base_argspec_path = os.path.join(self._role_path, 'meta', 'argument_specs')
    for ext in C.YAML_FILENAME_EXTENSIONS:
        full_path = base_argspec_path + ext
        if self._loader.path_exists(full_path):
            argument_specs = self._load_role_yaml('meta', main='argument_specs')
            try:
                return argument_specs.get('argument_specs') or {}
            except AttributeError:
                return {}
    return getattr(self._metadata, 'argument_specs', {})