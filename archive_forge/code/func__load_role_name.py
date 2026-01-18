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
def _load_role_name(self, ds):
    """
        Returns the role name (either the role: or name: field) from
        the role definition, or (when the role definition is a simple
        string), just that string
        """
    if isinstance(ds, string_types):
        return ds
    role_name = ds.get('role', ds.get('name'))
    if not role_name or not isinstance(role_name, string_types):
        raise AnsibleError('role definitions must contain a role name', obj=ds)
    if self._variable_manager:
        all_vars = self._variable_manager.get_vars(play=self._play)
        templar = Templar(loader=self._loader, variables=all_vars)
        role_name = templar.template(role_name)
    return role_name