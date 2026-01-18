from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareVersionInfo(_messages.Message):
    """Contains information about a specific Anthos on VMware version.

  Fields:
    dependencies: The list of upgrade dependencies for this version.
    hasDependencies: If set, the cluster dependencies (e.g. the admin cluster,
      other user clusters managed by the same admin cluster) must be upgraded
      before this version can be installed or upgraded to.
    isInstalled: If set, the version is installed in the admin cluster.
      Otherwise, the version bundle must be downloaded and installed before a
      user cluster can be created at or upgraded to this version.
    version: Version number e.g. 1.13.1-gke.1000.
  """
    dependencies = _messages.MessageField('UpgradeDependency', 1, repeated=True)
    hasDependencies = _messages.BooleanField(2)
    isInstalled = _messages.BooleanField(3)
    version = _messages.StringField(4)