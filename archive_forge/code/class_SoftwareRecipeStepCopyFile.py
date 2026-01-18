from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SoftwareRecipeStepCopyFile(_messages.Message):
    """Copies the artifact to the specified path on the instance.

  Fields:
    artifactId: Required. The id of the relevant artifact in the recipe.
    destination: Required. The absolute path on the instance to put the file.
    overwrite: Whether to allow this step to overwrite existing files. If this
      is false and the file already exists the file is not overwritten and the
      step is considered a success. Defaults to false.
    permissions: Consists of three octal digits which represent, in order, the
      permissions of the owner, group, and other users for the file (similarly
      to the numeric mode used in the linux chmod utility). Each digit
      represents a three bit number with the 4 bit corresponding to the read
      permissions, the 2 bit corresponds to the write bit, and the one bit
      corresponds to the execute permission. Default behavior is 755. Below
      are some examples of permissions and their associated values: read,
      write, and execute: 7 read and execute: 5 read and write: 6 read only: 4
  """
    artifactId = _messages.StringField(1)
    destination = _messages.StringField(2)
    overwrite = _messages.BooleanField(3)
    permissions = _messages.StringField(4)