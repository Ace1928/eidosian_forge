from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import json
from apitools.base.protorpclite import messages as protorpc_message
from apitools.base.py import encoding as protorpc_encoding
from googlecloudsdk.core.resource import resource_projection_parser
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import encoding
import six
from six.moves import range  # pylint: disable=redefined-builtin
class IdentityProjector(Projector):
    """A no-op resource projector that preserves the original object."""

    def __init__(self):
        super(IdentityProjector, self).__init__(resource_projection_parser.Parse())

    def Evaluate(self, obj):
        return obj