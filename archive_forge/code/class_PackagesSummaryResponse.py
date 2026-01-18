from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PackagesSummaryResponse(_messages.Message):
    """A summary of the packages found within the given resource.

  Fields:
    licensesSummary: A listing by license name of each of the licenses and
      their counts.
    resourceUrl: The unique URL of the image or the container for which this
      summary applies.
  """
    licensesSummary = _messages.MessageField('LicensesSummary', 1, repeated=True)
    resourceUrl = _messages.StringField(2)