from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CleanTextTag(_messages.Message):
    """Inspect text and transform sensitive text. Configurable using
  TextConfig. Supported [Value Representations] (http://dicom.nema.org/medical
  /dicom/2018e/output/chtml/part05/sect_6.2.html#table_6.2-1): AE, LO, LT, PN,
  SH, ST, UC, UT, DA, DT, AS
  """