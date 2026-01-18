from __future__ import (absolute_import, division, print_function)
import itertools
import operator
import os
from copy import copy as shallowcopy
from functools import cache
from jinja2.exceptions import UndefinedError
from ansible import constants as C
from ansible import context
from ansible.errors import AnsibleError, AnsibleParserError, AnsibleUndefinedVariable, AnsibleAssertionError
from ansible.module_utils.six import string_types
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.common.text.converters import to_text, to_native
from ansible.parsing.dataloader import DataLoader
from ansible.playbook.attribute import Attribute, FieldAttribute, ConnectionFieldAttribute, NonInheritableFieldAttribute
from ansible.plugins.loader import module_loader, action_loader
from ansible.utils.collection_loader._collection_finder import _get_collection_metadata, AnsibleCollectionRef
from ansible.utils.display import Display
from ansible.utils.sentinel import Sentinel
from ansible.utils.vars import combine_vars, isidentifier, get_unique_id
def _extend_value(self, value, new_value, prepend=False):
    """
        Will extend the value given with new_value (and will turn both
        into lists if they are not so already). The values are run through
        a set to remove duplicate values.
        """
    if not isinstance(value, list):
        value = [value]
    if not isinstance(new_value, list):
        new_value = [new_value]
    value = [v for v in value if v is not Sentinel]
    new_value = [v for v in new_value if v is not Sentinel]
    if prepend:
        combined = new_value + value
    else:
        combined = value + new_value
    return [i for i, dummy in itertools.groupby(combined) if i is not None]