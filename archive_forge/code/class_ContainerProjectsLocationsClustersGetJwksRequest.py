from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContainerProjectsLocationsClustersGetJwksRequest(_messages.Message):
    """A ContainerProjectsLocationsClustersGetJwksRequest object.

  Fields:
    parent: The cluster (project, location, cluster name) to get keys for.
      Specified in the format `projects/*/locations/*/clusters/*`.
  """
    parent = _messages.StringField(1, required=True)