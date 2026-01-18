from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import functools
from googlecloudsdk.calliope.concepts import deps as deps_lib
from googlecloudsdk.command_lib.concepts import base
from googlecloudsdk.command_lib.concepts import exceptions
from googlecloudsdk.command_lib.concepts import names
import six
class DependencyView(object):
    """Simple namespace used by concept.Parse for concept groups."""

    def __init__(self, values_dict):
        for key, value in six.iteritems(values_dict):
            setattr(self, names.ConvertToNamespaceName(key), value)