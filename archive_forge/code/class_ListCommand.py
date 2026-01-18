from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import contextlib
import enum
from functools import wraps  # pylint:disable=g-importing-member
import itertools
import re
import sys
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import display
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_printer
import six
class ListCommand(six.with_metaclass(abc.ABCMeta, CacheCommand)):
    """A command that pretty-prints all resources."""

    @staticmethod
    def _Flags(parser):
        """Adds the default flags for all ListCommand commands.

    Args:
      parser: The argparse parser.
    """
        FILTER_FLAG.AddToParser(parser)
        LIMIT_FLAG.AddToParser(parser)
        PAGE_SIZE_FLAG.AddToParser(parser)
        SORT_BY_FLAG.AddToParser(parser)
        URI_FLAG.AddToParser(parser)
        parser.display_info.AddFormat(properties.VALUES.core.default_format.Get())

    def Epilog(self, resources_were_displayed):
        """Called after resources are displayed if the default format was used.

    Args:
      resources_were_displayed: True if resources were displayed.
    """
        if not resources_were_displayed:
            log.status.Print('Listed 0 items.')