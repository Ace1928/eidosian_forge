from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WebServerNetworkAccessControl(_messages.Message):
    """Network-level access control policy for the Airflow web server.

  Fields:
    allowedIpRanges: A collection of allowed IP ranges with descriptions.
  """
    allowedIpRanges = _messages.MessageField('AllowedIpRange', 1, repeated=True)