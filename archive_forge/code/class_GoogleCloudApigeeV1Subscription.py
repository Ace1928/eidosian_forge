from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1Subscription(_messages.Message):
    """Pub/Sub subscription of an environment.

  Fields:
    name: Full name of the Pub/Sub subcription. Use the following structure in
      your request: `subscription "projects/foo/subscription/bar"`
  """
    name = _messages.StringField(1)