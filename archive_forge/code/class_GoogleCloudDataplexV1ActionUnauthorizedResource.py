from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1ActionUnauthorizedResource(_messages.Message):
    """Action details for unauthorized resource issues raised to indicate that
  the service account associated with the lake instance is not authorized to
  access or manage the resource associated with an asset.
  """