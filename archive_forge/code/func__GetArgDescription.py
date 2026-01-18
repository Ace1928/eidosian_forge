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
def _GetArgDescription(name, docstring_info):
    if docstring_info.args:
        for arg_in_docstring in docstring_info.args:
            if arg_in_docstring.name in (name, '*' + name, '**' + name):
                return arg_in_docstring.description
    return None