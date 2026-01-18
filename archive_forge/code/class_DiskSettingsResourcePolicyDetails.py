from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DiskSettingsResourcePolicyDetails(_messages.Message):
    """This is the object for storing the detail information about the Resource
  Policy that will be set as default ones for the Disks that is using the
  DiskSettings. It contains: - one target Resource Policy referenced by its
  Fully-Qualified URL, - [output only] Disk Types that will be excluded from
  using this Resource Policy, - Other filtering support (e.g. Label filtering)
  for Default Resource Policy can be added here as well

  Fields:
    excludedDiskTypes: [Output Only] A list of Disk Types that will be
      excluded from applying the Resource Policy referenced here. If absent,
      Disks created in any DiskType can use the referenced default Resource
      Policy.
    resourcePolicy: The target Resource Policies identified by their Fully-
      Qualified URL.
  """
    excludedDiskTypes = _messages.StringField(1, repeated=True)
    resourcePolicy = _messages.StringField(2)