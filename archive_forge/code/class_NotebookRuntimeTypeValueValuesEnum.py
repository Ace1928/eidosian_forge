from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotebookRuntimeTypeValueValuesEnum(_messages.Enum):
    """Optional. Immutable. The type of the notebook runtime template.

    Values:
      NOTEBOOK_RUNTIME_TYPE_UNSPECIFIED: Unspecified notebook runtime type,
        NotebookRuntimeType will default to USER_DEFINED.
      USER_DEFINED: runtime or template with coustomized configurations from
        user.
      ONE_CLICK: runtime or template with system defined configurations.
    """
    NOTEBOOK_RUNTIME_TYPE_UNSPECIFIED = 0
    USER_DEFINED = 1
    ONE_CLICK = 2