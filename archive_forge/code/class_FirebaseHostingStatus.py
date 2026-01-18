from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirebaseHostingStatus(_messages.Message):
    """Detailed status for Firebase Hosting resource.

  Fields:
    domains: List of domains associated with the firebase hosting site.
    hostingConfig: Hosting configuration created by Serverless Stacks.
  """
    domains = _messages.StringField(1, repeated=True)
    hostingConfig = _messages.StringField(2)