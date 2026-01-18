from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
class BinaryCommandFlag(six.with_metaclass(abc.ABCMeta, object)):
    """Informal interface for flags that get passed through to an underlying binary."""

    @abc.abstractmethod
    def AddToParser(self, parser):
        """Adds this argument to the given parser.

    Args:
      parser: The argparse parser.
    """
        pass

    @abc.abstractmethod
    def FormatFlags(self, args):
        """Return flags in a format that can be passed to the underlying binary."""
        pass