from __future__ import absolute_import, division, print_function
import ast
import json
import operator
import re
import socket
from copy import deepcopy
from functools import reduce  # forward compatibility for Python 3
from itertools import chain
from ansible.module_utils import basic
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.common._collections_compat import Mapping
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import iteritems, string_types
class ComplexList(EntityCollection):
    """Alternate name for EntityCollection for backwards compatibility"""

    def __init__(self, attrs, module, *args, **kwargs):
        super(ComplexList, self).__init__(module, attrs, *args, **kwargs)