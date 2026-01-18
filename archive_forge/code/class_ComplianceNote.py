from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComplianceNote(_messages.Message):
    """A ComplianceNote object.

  Fields:
    cisBenchmark: A CisBenchmark attribute.
    description: A description about this compliance check.
    impact: A string attribute.
    rationale: A rationale for the existence of this compliance check.
    remediation: A description of remediation steps if the compliance check
      fails.
    scanInstructions: Serialized scan instructions with a predefined format.
    title: The title that identifies this compliance check.
    version: The OS and config versions the benchmark applies to.
  """
    cisBenchmark = _messages.MessageField('CisBenchmark', 1)
    description = _messages.StringField(2)
    impact = _messages.StringField(3)
    rationale = _messages.StringField(4)
    remediation = _messages.StringField(5)
    scanInstructions = _messages.BytesField(6)
    title = _messages.StringField(7)
    version = _messages.MessageField('ComplianceVersion', 8, repeated=True)