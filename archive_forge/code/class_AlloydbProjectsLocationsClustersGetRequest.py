from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AlloydbProjectsLocationsClustersGetRequest(_messages.Message):
    """A AlloydbProjectsLocationsClustersGetRequest object.

  Enums:
    ViewValueValuesEnum: Optional. The view of the cluster to return. Returns
      all default fields if not set.

  Fields:
    name: Required. The name of the resource. For the required format, see the
      comment on the Cluster.name field.
    view: Optional. The view of the cluster to return. Returns all default
      fields if not set.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """Optional. The view of the cluster to return. Returns all default
    fields if not set.

    Values:
      CLUSTER_VIEW_UNSPECIFIED: CLUSTER_VIEW_UNSPECIFIED Not specified,
        equivalent to BASIC.
      CLUSTER_VIEW_BASIC: BASIC server responses include all the relevant
        cluster details, excluding
        Cluster.ContinuousBackupInfo.EarliestRestorableTime and other view-
        specific fields. The default value.
      CLUSTER_VIEW_CONTINUOUS_BACKUP: CONTINUOUS_BACKUP response returns all
        the fields from BASIC plus the earliest restorable time if continuous
        backups are enabled. May increase latency.
    """
        CLUSTER_VIEW_UNSPECIFIED = 0
        CLUSTER_VIEW_BASIC = 1
        CLUSTER_VIEW_CONTINUOUS_BACKUP = 2
    name = _messages.StringField(1, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 2)