from __future__ import (absolute_import, division, print_function)
import os
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleAssertionError
from ansible.module_utils.six import string_types
from ansible.parsing.yaml.objects import AnsibleBaseYAMLObject, AnsibleMapping
from ansible.playbook.attribute import NonInheritableFieldAttribute
from ansible.playbook.base import Base
from ansible.playbook.collectionsearch import CollectionSearch
from ansible.playbook.conditional import Conditional
from ansible.playbook.taggable import Taggable
from ansible.template import Templar
from ansible.utils.collection_loader import AnsibleCollectionRef
from ansible.utils.collection_loader._collection_finder import _get_collection_role_path
from ansible.utils.path import unfrackpath
from ansible.utils.display import Display
def _load_role_path(self, role_name):
    """
        the 'role', as specified in the ds (or as a bare string), can either
        be a simple name or a full path. If it is a full path, we use the
        basename as the role name, otherwise we take the name as-given and
        append it to the default role path
        """
    if self._variable_manager is not None:
        all_vars = self._variable_manager.get_vars(play=self._play)
    else:
        all_vars = dict()
    templar = Templar(loader=self._loader, variables=all_vars)
    role_name = templar.template(role_name)
    role_tuple = None
    if self._collection_list or AnsibleCollectionRef.is_valid_fqcr(role_name):
        role_tuple = _get_collection_role_path(role_name, self._collection_list)
    if role_tuple:
        self._role_collection = role_tuple[2]
        return role_tuple[0:2]
    role_search_paths = [os.path.join(self._loader.get_basedir(), u'roles')]
    if C.DEFAULT_ROLES_PATH:
        role_search_paths.extend(C.DEFAULT_ROLES_PATH)
    if self._role_basedir:
        role_search_paths.append(self._role_basedir)
    role_search_paths.append(self._loader.get_basedir())
    for path in role_search_paths:
        path = templar.template(path)
        role_path = unfrackpath(os.path.join(path, role_name))
        if self._loader.path_exists(role_path):
            return (role_name, role_path)
    role_path = unfrackpath(role_name)
    if self._loader.path_exists(role_path):
        role_name = os.path.basename(role_name)
        return (role_name, role_path)
    searches = (self._collection_list or []) + role_search_paths
    raise AnsibleError("the role '%s' was not found in %s" % (role_name, ':'.join(searches)), obj=self._ds)