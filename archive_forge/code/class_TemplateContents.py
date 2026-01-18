from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TemplateContents(_messages.Message):
    """Files that make up the template contents of a template type.

  Enums:
    InterpreterValueValuesEnum: Which interpreter (python or jinja) should be
      used during expansion.

  Fields:
    imports: Import files referenced by the main template.
    interpreter: Which interpreter (python or jinja) should be used during
      expansion.
    mainTemplate: The filename of the mainTemplate
    schema: The contents of the template schema.
    template: The contents of the main template file.
  """

    class InterpreterValueValuesEnum(_messages.Enum):
        """Which interpreter (python or jinja) should be used during expansion.

    Values:
      UNKNOWN_INTERPRETER: <no description>
      PYTHON: <no description>
      JINJA: <no description>
    """
        UNKNOWN_INTERPRETER = 0
        PYTHON = 1
        JINJA = 2
    imports = _messages.MessageField('ImportFile', 1, repeated=True)
    interpreter = _messages.EnumField('InterpreterValueValuesEnum', 2)
    mainTemplate = _messages.StringField(3)
    schema = _messages.StringField(4)
    template = _messages.StringField(5)