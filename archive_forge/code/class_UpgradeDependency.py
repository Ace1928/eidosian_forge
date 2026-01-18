from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpgradeDependency(_messages.Message):
    """UpgradeDependency represents a dependency when upgrading a resource.

  Fields:
    currentVersion: Current version of the dependency e.g. 1.15.0.
    membership: Membership names are formatted as
      `projects//locations//memberships/`.
    resourceName: Resource name of the dependency.
    targetVersion: Target version of the dependency e.g. 1.16.1. This is the
      version the dependency needs to be upgraded to before a resource can be
      upgraded.
  """
    currentVersion = _messages.StringField(1)
    membership = _messages.StringField(2)
    resourceName = _messages.StringField(3)
    targetVersion = _messages.StringField(4)