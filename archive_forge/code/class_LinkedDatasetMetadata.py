from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LinkedDatasetMetadata(_messages.Message):
    """Metadata about the Linked Dataset.

  Enums:
    LinkStateValueValuesEnum: Output only. Specifies whether Linked Dataset is
      currently in a linked state or not.

  Fields:
    linkState: Output only. Specifies whether Linked Dataset is currently in a
      linked state or not.
  """

    class LinkStateValueValuesEnum(_messages.Enum):
        """Output only. Specifies whether Linked Dataset is currently in a linked
    state or not.

    Values:
      LINK_STATE_UNSPECIFIED: The default value. Default to the LINKED state.
      LINKED: Normal Linked Dataset state. Data is queryable via the Linked
        Dataset.
      UNLINKED: Data publisher or owner has unlinked this Linked Dataset. It
        means you can no longer query or see the data in the Linked Dataset.
    """
        LINK_STATE_UNSPECIFIED = 0
        LINKED = 1
        UNLINKED = 2
    linkState = _messages.EnumField('LinkStateValueValuesEnum', 1)