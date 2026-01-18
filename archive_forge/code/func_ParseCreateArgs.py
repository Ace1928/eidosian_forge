from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
import six
def ParseCreateArgs(args, labels_cls, labels_dest='labels'):
    """Initializes labels based on args and the given class."""
    labels = getattr(args, labels_dest)
    if labels is None:
        return None
    return _PackageLabels(labels_cls, labels)