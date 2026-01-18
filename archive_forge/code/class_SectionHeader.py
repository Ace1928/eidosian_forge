from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SectionHeader(_messages.Message):
    """A widget that defines a new section header. Sections populate a table of
  contents and allow easier navigation of long-form content.

  Fields:
    dividerBelow: Whether to insert a divider below the section in the table
      of contents
    subtitle: The subtitle of the section
  """
    dividerBelow = _messages.BooleanField(1)
    subtitle = _messages.StringField(2)