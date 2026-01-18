from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpgradeOccurrence(_messages.Message):
    """An Upgrade Occurrence represents that a specific resource_url could
  install a specific upgrade. This presence is supplied via local sources
  (i.e. it is present in the mirror and the running system has noticed its
  availability). For Windows, both distribution and windows_update contain
  information for the Windows update.

  Fields:
    distribution: Metadata about the upgrade for available for the specific
      operating system for the resource_url. This allows efficient filtering,
      as well as making it easier to use the occurrence.
    package: Required for non-Windows OS. The package this Upgrade is for.
    parsedVersion: Required for non-Windows OS. The version of the package in
      a machine + human readable form.
    windowsUpdate: Required for Windows OS. Represents the metadata about the
      Windows update.
  """
    distribution = _messages.MessageField('UpgradeDistribution', 1)
    package = _messages.StringField(2)
    parsedVersion = _messages.MessageField('Version', 3)
    windowsUpdate = _messages.MessageField('WindowsUpdate', 4)