from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2DocumentLocation(_messages.Message):
    """Location of a finding within a document.

  Fields:
    fileOffset: Offset of the line, from the beginning of the file, where the
      finding is located.
  """
    fileOffset = _messages.IntegerField(1)