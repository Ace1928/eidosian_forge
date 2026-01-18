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
def _GetSummary(info):
    docstring_info = info['docstring_info']
    return docstring_info.summary if docstring_info.summary else None