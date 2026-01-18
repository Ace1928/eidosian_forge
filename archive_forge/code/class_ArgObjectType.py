from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
import re
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import util as format_util
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import http_encoding
import six
class ArgObjectType(object):
    """An interface for custom type generators that bind directly to a message.

  Like ArgDict, ArgObject type can only be generated one we know the type
  of the message.
  """

    def GenerateType(self, field):
        """Generates an argparse type function to use to parse the argument.

    Args:
      field: The apitools field instance.
    """

    def Action(self, unused_repeated):
        """The argparse action to use for this argument.

    'store' is the default action, but sometimes something like 'append' might
    be required to allow the argument to be repeated and all values collected.

    Args:
      unused_repeated: whether or not the message is repeated

    Returns:
      str, The argparse action to use.
    """
        return 'store'