from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
class _IdentityIdResolver(IdResolver):
    """An IdResolver that resolve app_id == project_id."""

    def resolve_project_id(self, app_id):
        return app_id

    def resolve_app_id(self, project_id):
        return project_id