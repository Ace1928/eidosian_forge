from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OSPolicyResourcePackageResourceDeb(_messages.Message):
    """A deb package file. dpkg packages only support INSTALLED state.

  Fields:
    pullDeps: Whether dependencies should also be installed. - install when
      false: `dpkg -i package` - install when true: `apt-get update && apt-get
      -y install package.deb`
    source: Required. A deb package.
  """
    pullDeps = _messages.BooleanField(1)
    source = _messages.MessageField('OSPolicyResourceFile', 2)