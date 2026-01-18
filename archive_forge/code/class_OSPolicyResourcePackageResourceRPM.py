from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OSPolicyResourcePackageResourceRPM(_messages.Message):
    """An RPM package file. RPM packages only support INSTALLED state.

  Fields:
    pullDeps: Whether dependencies should also be installed. - install when
      false: `rpm --upgrade --replacepkgs package.rpm` - install when true:
      `yum -y install package.rpm` or `zypper -y install package.rpm`
    source: Required. An rpm package.
  """
    pullDeps = _messages.BooleanField(1)
    source = _messages.MessageField('OSPolicyResourceFile', 2)