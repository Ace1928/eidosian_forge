from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import os
import re
import textwrap
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_projector
from googlecloudsdk.core.resource import yaml_printer
from googlecloudsdk.core.util import files
import six
def _NormalizeTypeName(name):
    """Returns the JSON schema normalized type name for name."""
    s = six.text_type(name).lower()
    if re.match('.?int64', s):
        return 'integer'
    if re.match('.?int32', s):
        return 'integer'
    if re.match('^int\\d*$', s):
        return 'integer'
    if s == 'float':
        return 'number'
    if s == 'double':
        return 'number'
    if s == 'bool':
        return 'boolean'
    if s == 'bytes':
        return 'string'
    return s