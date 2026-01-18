from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.core import exceptions
import six
class FlagDef(object):
    """Object that holds a flag definition and adds it to a parser."""

    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs

    def __eq__(self, other):
        return self.name == other.name

    def __ne__(self, other):
        return self.name != other.name

    def __hash__(self):
        return hash(self.name)

    def ConfigureParser(self, parser):
        parser.add_argument(self.name, **self.kwargs)