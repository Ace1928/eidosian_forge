from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from googlecloudsdk.api_lib.dataflow import exceptions
from googlecloudsdk.core.util import files
import six
def _IsString(val):
    try:
        return isinstance(val, unicode)
    except NameError:
        return isinstance(val, str)