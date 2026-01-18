from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EndpointMatcher(_messages.Message):
    """A definition of a matcher that selects endpoints to which the policies
  should be applied.

  Fields:
    metadataLabelMatcher: The matcher is based on node metadata presented by
      xDS clients.
  """
    metadataLabelMatcher = _messages.MessageField('EndpointMatcherMetadataLabelMatcher', 1)