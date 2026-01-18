from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2Reference(_messages.Message):
    """Additional Links

  Fields:
    source: Source of the reference e.g. NVD
    uri: Uri for the mentioned source e.g. https://cve.mitre.org/cgi-
      bin/cvename.cgi?name=CVE-2021-34527.
  """
    source = _messages.StringField(1)
    uri = _messages.StringField(2)