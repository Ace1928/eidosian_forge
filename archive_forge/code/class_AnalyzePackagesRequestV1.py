from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnalyzePackagesRequestV1(_messages.Message):
    """AnalyzePackagesRequest is the request to analyze a list of packages and
  create Vulnerability Occurrences for it.

  Fields:
    includeOsvData: [DEPRECATED] Whether to include OSV data in the scan. For
      backwards compatibility reasons, this field can be neither removed nor
      renamed.
    packages: The packages to analyze.
    resourceUri: Required. The resource URI of the container image being
      scanned.
  """
    includeOsvData = _messages.BooleanField(1)
    packages = _messages.MessageField('PackageData', 2, repeated=True)
    resourceUri = _messages.StringField(3)