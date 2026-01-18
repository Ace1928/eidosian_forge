from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContainerProjectsZonesGetServerconfigRequest(_messages.Message):
    """A ContainerProjectsZonesGetServerconfigRequest object.

  Fields:
    name: The name (project and location) of the server config to get,
      specified in the format `projects/*/locations/*`.
    projectId: Deprecated. The Google Developers Console [project ID or
      project number](https://cloud.google.com/resource-manager/docs/creating-
      managing-projects). This field has been deprecated and replaced by the
      name field.
    zone: Deprecated. The name of the Google Compute Engine
      [zone](https://cloud.google.com/compute/docs/zones#available) to return
      operations for. This field has been deprecated and replaced by the name
      field.
  """
    name = _messages.StringField(1)
    projectId = _messages.StringField(2, required=True)
    zone = _messages.StringField(3, required=True)