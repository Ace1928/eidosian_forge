from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OSPolicyResourceFileResource(_messages.Message):
    """A resource that manages the state of a file.

  Enums:
    StateValueValuesEnum: Required. Desired state of the file.

  Fields:
    content: A a file with this content. The size of the content is limited to
      32KiB.
    file: A remote or local source.
    path: Required. The absolute path of the file within the VM.
    permissions: Consists of three octal digits which represent, in order, the
      permissions of the owner, group, and other users for the file (similarly
      to the numeric mode used in the linux chmod utility). Each digit
      represents a three bit number with the 4 bit corresponding to the read
      permissions, the 2 bit corresponds to the write bit, and the one bit
      corresponds to the execute permission. Default behavior is 755. Below
      are some examples of permissions and their associated values: read,
      write, and execute: 7 read and execute: 5 read and write: 6 read only: 4
    state: Required. Desired state of the file.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Required. Desired state of the file.

    Values:
      DESIRED_STATE_UNSPECIFIED: Unspecified is invalid.
      PRESENT: Ensure file at path is present.
      ABSENT: Ensure file at path is absent.
      CONTENTS_MATCH: Ensure the contents of the file at path matches. If the
        file does not exist it will be created.
    """
        DESIRED_STATE_UNSPECIFIED = 0
        PRESENT = 1
        ABSENT = 2
        CONTENTS_MATCH = 3
    content = _messages.StringField(1)
    file = _messages.MessageField('OSPolicyResourceFile', 2)
    path = _messages.StringField(3)
    permissions = _messages.StringField(4)
    state = _messages.EnumField('StateValueValuesEnum', 5)