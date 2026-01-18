from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1RoutineSpecArgument(_messages.Message):
    """Input or output argument of a function or stored procedure.

  Enums:
    ModeValueValuesEnum: Specifies whether the argument is input or output.

  Fields:
    mode: Specifies whether the argument is input or output.
    name: The name of the argument. A return argument of a function might not
      have a name.
    type: Type of the argument. The exact value depends on the source system
      and the language.
  """

    class ModeValueValuesEnum(_messages.Enum):
        """Specifies whether the argument is input or output.

    Values:
      MODE_UNSPECIFIED: Unspecified mode.
      IN: The argument is input-only.
      OUT: The argument is output-only.
      INOUT: The argument is both an input and an output.
    """
        MODE_UNSPECIFIED = 0
        IN = 1
        OUT = 2
        INOUT = 3
    mode = _messages.EnumField('ModeValueValuesEnum', 1)
    name = _messages.StringField(2)
    type = _messages.StringField(3)