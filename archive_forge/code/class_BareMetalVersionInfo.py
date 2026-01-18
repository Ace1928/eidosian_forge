from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalVersionInfo(_messages.Message):
    """Contains information about a specific Anthos on bare metal version.

  Fields:
    dependencies: The list of upgrade dependencies for this version.
    hasDependencies: If set, the cluster dependencies (e.g. the admin cluster,
      other user clusters managed by the same admin cluster, version skew
      policy, etc) must be upgraded before this version can be installed or
      upgraded to.
    version: Version number e.g. 1.13.1.
  """
    dependencies = _messages.MessageField('UpgradeDependency', 1, repeated=True)
    hasDependencies = _messages.BooleanField(2)
    version = _messages.StringField(3)