from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CustomModuleValidationError(_messages.Message):
    """An error encountered while validating the uploaded configuration of an
  Event Threat Detection Custom Module.

  Fields:
    description: A description of the error, suitable for human consumption.
      Required.
    end: The end position of the error in the uploaded text version of the
      module. This field may be omitted if no specific position applies, or if
      one could not be computed..
    fieldPath: The path, in RFC 8901 JSON Pointer format, to the field that
      failed validation. This may be left empty if no specific field is
      affected.
    start: The initial position of the error in the uploaded text version of
      the module. This field may be omitted if no specific position applies,
      or if one could not be computed.
  """
    description = _messages.StringField(1)
    end = _messages.MessageField('Position', 2)
    fieldPath = _messages.StringField(3)
    start = _messages.MessageField('Position', 4)