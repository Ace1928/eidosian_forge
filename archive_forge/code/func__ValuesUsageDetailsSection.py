from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import itertools
import sys
from fire import completion
from fire import custom_descriptions
from fire import decorators
from fire import docstrings
from fire import formatting
from fire import inspectutils
from fire import value_types
def _ValuesUsageDetailsSection(component, values):
    """Creates a section tuple for the values section of the usage details."""
    value_item_strings = []
    for value_name, value in values.GetItems():
        del value
        init_info = inspectutils.Info(component.__class__.__init__)
        value_item = None
        if 'docstring_info' in init_info:
            init_docstring_info = init_info['docstring_info']
            if init_docstring_info.args:
                for arg_info in init_docstring_info.args:
                    if arg_info.name == value_name:
                        value_item = _CreateItem(value_name, arg_info.description)
        if value_item is None:
            value_item = str(value_name)
        value_item_strings.append(value_item)
    return ('VALUES', _NewChoicesSection('VALUE', value_item_strings))