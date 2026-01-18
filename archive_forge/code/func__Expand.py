from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import json
import logging
import string
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.util import resource as resource_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import encoding
import six
def _Expand(s):
    """Return s with escaped format special characters expanded."""
    r = []
    n = 0
    i = 0
    while i < len(s):
        c = s[i]
        i += 1
        if c == _ESCAPE and i + 1 < len(s) and (s[i + 1] == _ESCAPE):
            c = s[i]
            i += 2
            if c == _ESCAPED_LEFT_CURLY:
                if n > 0:
                    r.append(_ESCAPE + _ESCAPED_LEFT_CURLY)
                else:
                    r.append('{')
                n += 1
            elif c == _ESCAPED_RIGHT_CURLY:
                n -= 1
                if n > 0:
                    r.append(_ESCAPE + _ESCAPED_RIGHT_CURLY)
                else:
                    r.append('}')
            elif n > 0:
                r.append(s[i - 3:i])
            elif c == _ESCAPED_COLON:
                r.append(':')
            elif c == _ESCAPED_ESCAPE:
                r.append(_ESCAPE)
        else:
            r.append(c)
    return ''.join(r)