from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComplianceVersion(_messages.Message):
    """Describes the CIS benchmark version that is applicable to a given OS and
  os version.

  Fields:
    benchmarkDocument: The name of the document that defines this benchmark,
      e.g. "CIS Container-Optimized OS".
    cpeUri: The CPE URI (https://cpe.mitre.org/specification/) this benchmark
      is applicable to.
    version: The version of the benchmark. This is set to the version of the
      OS-specific CIS document the benchmark is defined in.
  """
    benchmarkDocument = _messages.StringField(1)
    cpeUri = _messages.StringField(2)
    version = _messages.StringField(3)