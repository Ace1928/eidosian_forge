from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContainerProjectsLocationsOperationsGetRequest(_messages.Message):
    """A ContainerProjectsLocationsOperationsGetRequest object.

  Fields:
    name: The name (project, location, operation id) of the operation to get.
      Specified in the format `projects/*/locations/*/operations/*`.
    operationId: Deprecated. The server-assigned `name` of the operation. This
      field has been deprecated and replaced by the name field.
    projectId: Deprecated. The Google Developers Console [project ID or
      project number](https://cloud.google.com/resource-manager/docs/creating-
      managing-projects). This field has been deprecated and replaced by the
      name field.
    zone: Deprecated. The name of the Google Compute Engine
      [zone](https://cloud.google.com/compute/docs/zones#available) in which
      the cluster resides. This field has been deprecated and replaced by the
      name field.
  """
    name = _messages.StringField(1, required=True)
    operationId = _messages.StringField(2)
    projectId = _messages.StringField(3)
    zone = _messages.StringField(4)