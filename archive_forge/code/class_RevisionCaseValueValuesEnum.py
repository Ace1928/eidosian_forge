from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RevisionCaseValueValuesEnum(_messages.Enum):
    """Reads the revision by the predefined case.

    Values:
      REVISION_CASE_UNSPECIFIED: Unspecified case, fall back to read the
        `LATEST_HUMAN_REVIEW`.
      LATEST_HUMAN_REVIEW: The latest revision made by a human.
      LATEST_TIMESTAMP: The latest revision based on timestamp.
      BASE_OCR_REVISION: The first (OCR) revision.
    """
    REVISION_CASE_UNSPECIFIED = 0
    LATEST_HUMAN_REVIEW = 1
    LATEST_TIMESTAMP = 2
    BASE_OCR_REVISION = 3