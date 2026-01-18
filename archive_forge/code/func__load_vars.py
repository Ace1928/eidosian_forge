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
def _load_vars(self, attr, ds):
    """
        Vars in a play can be specified either as a dictionary directly, or
        as a list of dictionaries. If the later, this method will turn the
        list into a single dictionary.
        """

    def _validate_variable_keys(ds):
        for key in ds:
            if not isidentifier(key):
                raise TypeError("'%s' is not a valid variable name" % key)
    try:
        if isinstance(ds, dict):
            _validate_variable_keys(ds)
            return combine_vars(self.vars, ds)
        elif isinstance(ds, list):
            display.deprecated('Specifying a list of dictionaries for vars is deprecated in favor of specifying a dictionary.', version='2.18')
            all_vars = self.vars
            for item in ds:
                if not isinstance(item, dict):
                    raise ValueError
                _validate_variable_keys(item)
                all_vars = combine_vars(all_vars, item)
            return all_vars
        elif ds is None:
            return {}
        else:
            raise ValueError
    except ValueError as e:
        raise AnsibleParserError('Vars in a %s must be specified as a dictionary' % self.__class__.__name__, obj=ds, orig_exc=e)
    except TypeError as e:
        raise AnsibleParserError('Invalid variable name in vars specified for %s: %s' % (self.__class__.__name__, e), obj=ds, orig_exc=e)