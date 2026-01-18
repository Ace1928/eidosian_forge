from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeidentifyOperationMetadata(_messages.Message):
    """Details about the work the de-identify operation performed.

  Fields:
    fhirOutput: Details about the FHIR store to write the output to.
  """
    fhirOutput = _messages.MessageField('FhirOutput', 1)