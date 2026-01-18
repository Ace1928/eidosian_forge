from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AppengineAppsServicesPatchRequest(_messages.Message):
    """A AppengineAppsServicesPatchRequest object.

  Fields:
    migrateTraffic: Set to true to gradually shift traffic to one or more
      versions that you specify. By default, traffic is shifted immediately.
      For gradual traffic migration, the target versions must be located
      within instances that are configured for both warmup requests
      (https://cloud.google.com/appengine/docs/admin-
      api/reference/rest/v1beta/apps.services.versions#InboundServiceType) and
      automatic scaling (https://cloud.google.com/appengine/docs/admin-
      api/reference/rest/v1beta/apps.services.versions#AutomaticScaling). You
      must specify the shardBy (https://cloud.google.com/appengine/docs/admin-
      api/reference/rest/v1beta/apps.services#ShardBy) field in the Service
      resource. Gradual traffic migration is not supported in the App Engine
      flexible environment. For examples, see Migrating and Splitting Traffic
      (https://cloud.google.com/appengine/docs/admin-api/migrating-splitting-
      traffic).
    name: Name of the resource to update. Example:
      apps/myapp/services/default.
    service: A Service resource to be passed as the request body.
    updateMask: Required. Standard field mask for the set of fields to be
      updated.
  """
    migrateTraffic = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)
    service = _messages.MessageField('Service', 3)
    updateMask = _messages.StringField(4)