from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SetLegacyAbacRequest(_messages.Message):
    """SetLegacyAbacRequest enables or disables the ABAC authorization
  mechanism for a cluster.

  Fields:
    clusterId: Deprecated. The name of the cluster to update. This field has
      been deprecated and replaced by the name field.
    enabled: Required. Whether ABAC authorization will be enabled in the
      cluster.
    name: The name (project, location, cluster name) of the cluster to set
      legacy abac. Specified in the format
      `projects/*/locations/*/clusters/*`.
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
    enabled = _messages.BooleanField(2)
    name = _messages.StringField(3)
    projectId = _messages.StringField(4)
    zone = _messages.StringField(5)