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
def _load_role_yaml(self, subdir, main=None, allow_dir=False):
    """
        Find and load role YAML files and return data found.
        :param subdir: subdir of role to search (vars, files, tasks, handlers, defaults)
        :type subdir: string
        :param main: filename to match, will default to 'main.<ext>' if not provided.
        :type main: string
        :param allow_dir: If true we combine results of multiple matching files found.
                          If false, highlander rules. Only for vars(dicts) and not tasks(lists).
        :type allow_dir: bool

        :returns: data from the matched file(s), type can be dict or list depending on vars or tasks.
        """
    data = None
    file_path = os.path.join(self._role_path, subdir)
    if self._loader.path_exists(file_path) and self._loader.is_directory(file_path):
        extensions = ['.yml', '.yaml', '.json']
        if main is None:
            _main = 'main'
            extensions.append('')
        else:
            _main = main
            extensions.insert(0, '')
        found_files = self._loader.find_vars_files(file_path, _main, extensions, allow_dir)
        if found_files:
            for found in found_files:
                if not is_subpath(found, file_path):
                    raise AnsibleParserError("Failed loading '%s' for role (%s) as it is not inside the expected role path: '%s'" % (to_text(found), self._role_name, to_text(file_path)))
                new_data = self._loader.load_from_file(found)
                if new_data:
                    if data is not None and isinstance(new_data, Mapping):
                        data = combine_vars(data, new_data)
                    else:
                        data = new_data
                    if not allow_dir:
                        break
        elif main is not None:
            raise AnsibleParserError('Could not find specified file in role: %s/%s' % (subdir, main))
    return data