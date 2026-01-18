from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.metastore import validators as validator
import six
def _AppendKeysToUpdateMask(prefix, key):
    return prefix + '.' + key