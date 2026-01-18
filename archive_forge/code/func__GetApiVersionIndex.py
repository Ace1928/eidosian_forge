from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import exceptions
def _GetApiVersionIndex(tokens):
    for idx, token in enumerate(tokens):
        if IsApiVersion(token):
            return idx
    return -1