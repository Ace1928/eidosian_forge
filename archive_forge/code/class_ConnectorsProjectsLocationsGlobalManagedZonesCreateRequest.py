from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConnectorsProjectsLocationsGlobalManagedZonesCreateRequest(_messages.Message):
    """A ConnectorsProjectsLocationsGlobalManagedZonesCreateRequest object.

  Fields:
    managedZone: A ManagedZone resource to be passed as the request body.
    managedZoneId: Required. Identifier to assign to the ManagedZone. Must be
      unique within scope of the parent resource.
    parent: Required. Parent resource of the ManagedZone, of the form:
      `projects/*/locations/global`
  """
    managedZone = _messages.MessageField('ManagedZone', 1)
    managedZoneId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)