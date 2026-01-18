from __future__ import absolute_import
from __future__ import unicode_literals
import base64
import collections
import pickle
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine._internal import six_subset
from googlecloudsdk.third_party.appengine.api import datastore_errors
from googlecloudsdk.third_party.appengine.api import datastore_types
from googlecloudsdk.third_party.appengine.datastore import datastore_index
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_pbs
from googlecloudsdk.third_party.appengine.datastore import datastore_rpc
def _validate_projection_and_group_by(self, projection, group_by):
    """Validates that a query's projection and group by match.

    Args:
      projection: A set of string property names in the projection.
      group_by: A set of string property names in the group by.
    Raises:
      datastore_errors.BadRequestError: if the projection and group
        by sets are not equal.
    """
    if projection:
        if group_by:
            extra = set(projection) - set(group_by)
            if extra:
                raise datastore_errors.BadRequestError('projections includes properties not in the group_by argument: %s' % extra)
    elif group_by:
        raise datastore_errors.BadRequestError('cannot specify group_by without a projection')