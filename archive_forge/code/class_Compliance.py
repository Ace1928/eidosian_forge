from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Compliance(_messages.Message):
    """Contains compliance information about a security standard indicating
  unmet recommendations.

  Fields:
    ids: Policies within the standard or benchmark, for example, A.12.4.1
    standard: Industry-wide compliance standards or benchmarks, such as CIS,
      PCI, and OWASP.
    version: Version of the standard or benchmark, for example, 1.1
  """
    ids = _messages.StringField(1, repeated=True)
    standard = _messages.StringField(2)
    version = _messages.StringField(3)