from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LicensesSummary(_messages.Message):
    """Per license count

  Fields:
    count: The number of fixable vulnerabilities associated with this
      resource.
    license: The license of the package. Note that the format of this value is
      not guaranteed. It may be nil, an empty string, a boolean value (A | B),
      a differently formed boolean value (A OR B), etc...
  """
    count = _messages.IntegerField(1)
    license = _messages.StringField(2)