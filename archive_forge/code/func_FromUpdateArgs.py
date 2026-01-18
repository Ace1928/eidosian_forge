from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
import six
@classmethod
def FromUpdateArgs(cls, args, enable_clear=True):
    """Initializes a Diff based on the arguments in AddUpdateLabelsFlags."""
    if enable_clear:
        clear = args.clear_labels
    else:
        clear = None
    return cls(args.update_labels, args.remove_labels, clear)