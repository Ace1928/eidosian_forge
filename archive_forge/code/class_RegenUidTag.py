from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RegenUidTag(_messages.Message):
    """Replace UID with a new generated UID. Supported [Value Representation] (
  http://dicom.nema.org/medical/dicom/2018e/output/chtml/part05/sect_6.2.html#
  table_6.2-1): UI
  """