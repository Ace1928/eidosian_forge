from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ScriptStackFrame(_messages.Message):
    """Represents the location of the statement/expression being evaluated.
  Line and column numbers are defined as follows: - Line and column numbers
  start with one. That is, line 1 column 1 denotes the start of the script. -
  When inside a stored procedure, all line/column numbers are relative to the
  procedure body, not the script in which the procedure was defined. -
  Start/end positions exclude leading/trailing comments and whitespace. The
  end position always ends with a ";", when present. - Multi-byte Unicode
  characters are treated as just one column. - If the original script (or
  procedure definition) contains TAB characters, a tab "snaps" the indentation
  forward to the nearest multiple of 8 characters, plus 1. For example, a TAB
  on column 1, 2, 3, 4, 5, 6 , or 8 will advance the next character to column
  9. A TAB on column 9, 10, 11, 12, 13, 14, 15, or 16 will advance the next
  character to column 17.

  Fields:
    endColumn: Output only. One-based end column.
    endLine: Output only. One-based end line.
    procedureId: Output only. Name of the active procedure, empty if in a top-
      level script.
    startColumn: Output only. One-based start column.
    startLine: Output only. One-based start line.
    text: Output only. Text of the current statement/expression.
  """
    endColumn = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    endLine = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    procedureId = _messages.StringField(3)
    startColumn = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    startLine = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    text = _messages.StringField(6)