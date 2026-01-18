from __future__ import (absolute_import, division, print_function)
import os
from collections.abc import Mapping, Sequence
from jinja2.exceptions import UndefinedError
from ansible.errors import AnsibleLookupError, AnsibleUndefinedVariable
from ansible.module_utils.six import string_types
from ansible.plugins.lookup import LookupBase
def _split_on(terms, spliters=','):
    termlist = []
    if isinstance(terms, string_types):
        termlist = list(_splitter(terms, spliters))
    else:
        for t in terms:
            termlist.extend(_split_on(t, spliters))
    return termlist