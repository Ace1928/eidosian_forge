from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnalyzePackagesResponse(_messages.Message):
    """AnalyzePackagesResponse contains the information necessary to find
  results for the given scan.

  Fields:
    scan: The name of the scan resource created by this successful scan.
  """
    scan = _messages.StringField(1)