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
def _CreateItem(name, description, indent=2):
    if not description:
        return name
    return '{name}\n{description}'.format(name=name, description=formatting.Indent(description, indent))