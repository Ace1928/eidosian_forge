from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.core import exceptions
import six
def ConfigureParser(self, parser):
    group = parser.add_mutually_exclusive_group(required=False)
    for op in self._operations:
        op.ConfigureParser(group)