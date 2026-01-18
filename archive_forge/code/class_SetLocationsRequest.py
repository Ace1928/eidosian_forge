from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SetLocationsRequest(_messages.Message):
    """SetLocationsRequest sets the locations of the cluster.

  Fields:
    clusterId: Deprecated. The name of the cluster to upgrade. This field has
      been deprecated and replaced by the name field.
    locations: Required. The desired list of Google Compute Engine
      [zones](https://cloud.google.com/compute/docs/zones#available) in which
      the cluster's nodes should be located. Changing the locations a cluster
      is in will result in nodes being either created or removed from the
      cluster, depending on whether locations are being added or removed. This
      list must always include the cluster's primary zone.
    name: The name (project, location, cluster) of the cluster to set
      locations. Specified in the format `projects/*/locations/*/clusters/*`.
    projectId: Deprecated. The Google Developers Console [project ID or
      project number](https://cloud.google.com/resource-manager/docs/creating-
      managing-projects). This field has been deprecated and replaced by the
      name field.
    zone: Deprecated. The name of the Google Compute Engine
      [zone](https://cloud.google.com/compute/docs/zones#available) in which
      the cluster resides. This field has been deprecated and replaced by the
      name field.
  """
    clusterId = _messages.StringField(1)
    locations = _messages.StringField(2, repeated=True)
    name = _messages.StringField(3)
    projectId = _messages.StringField(4)
    zone = _messages.StringField(5)