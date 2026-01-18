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
def _split_role_params(self, ds):
    """
        Splits any random role params off from the role spec and store
        them in a dictionary of params for parsing later
        """
    role_def = dict()
    role_params = dict()
    base_attribute_names = frozenset(self.fattributes)
    for key, value in ds.items():
        if key not in base_attribute_names:
            role_params[key] = value
        else:
            role_def[key] = value
    return (role_def, role_params)