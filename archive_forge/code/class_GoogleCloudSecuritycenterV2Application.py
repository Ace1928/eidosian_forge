from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2Application(_messages.Message):
    """Represents an application associated with a finding.

  Fields:
    baseUri: The base URI that identifies the network location of the
      application in which the vulnerability was detected. For example,
      `http://example.com`.
    fullUri: The full URI with payload that could be used to reproduce the
      vulnerability. For example, `http://example.com?p=aMmYgI6H`.
  """
    baseUri = _messages.StringField(1)
    fullUri = _messages.StringField(2)