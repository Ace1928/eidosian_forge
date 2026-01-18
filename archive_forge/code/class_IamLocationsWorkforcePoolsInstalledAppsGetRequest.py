from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamLocationsWorkforcePoolsInstalledAppsGetRequest(_messages.Message):
    """A IamLocationsWorkforcePoolsInstalledAppsGetRequest object.

  Fields:
    name: Required. The name of the workforce pool installed app to retrieve.
      Format: `locations/{location}/workforcePools/{workforce_pool}/installedA
      pps/{installed_app}`
  """
    name = _messages.StringField(1, required=True)