from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OSPolicyResourcePackageResourceYUM(_messages.Message):
    """A package managed by YUM. - install: `yum -y install package` - remove:
  `yum -y remove package`

  Fields:
    name: Required. Package name.
  """
    name = _messages.StringField(1)