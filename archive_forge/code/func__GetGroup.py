from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.core import exceptions
import six
def _GetGroup(self, klass):
    if klass not in self._group_cache:
        group = klass()
        self._group_cache[klass] = group
        self._AddOperation(group)
    return self._group_cache[klass]