from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnalyzePackagesRequest(_messages.Message):
    """AnalyzePackagesRequest is the request to analyze a list of packages and
  create Vulnerability Occurrences for it.

  Fields:
    packages: The packages to analyze.
    resourceUri: Required. The resource URI of the container image being
      scanned.
  """
    packages = _messages.MessageField('PackageData', 1, repeated=True)
    resourceUri = _messages.StringField(2)