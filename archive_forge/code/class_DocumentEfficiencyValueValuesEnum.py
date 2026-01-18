from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DocumentEfficiencyValueValuesEnum(_messages.Enum):
    """Optional. Whether or not the suggested document is efficient. For
    example, if the document is poorly written, hard to understand, hard to
    use or too long to find useful information, document_efficiency is
    DocumentEfficiency.INEFFICIENT.

    Values:
      DOCUMENT_EFFICIENCY_UNSPECIFIED: Document efficiency unspecified.
      INEFFICIENT: Document is inefficient.
      EFFICIENT: Document is efficient.
    """
    DOCUMENT_EFFICIENCY_UNSPECIFIED = 0
    INEFFICIENT = 1
    EFFICIENT = 2