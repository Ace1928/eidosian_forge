from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1Fleet(_messages.Message):
    """Fleet related configuration. Fleets are a Google Cloud concept for
  logically organizing clusters, letting you use and manage multi-cluster
  capabilities and apply consistent policies across your systems. See [Anthos
  Fleets](https://cloud.google.com/anthos/multicluster-management/fleets) for
  more details on Anthos multi-cluster capabilities using Fleets.

  Fields:
    membership: Output only. The name of the managed Hub Membership resource
      associated to this cluster. Membership names are formatted as
      `projects//locations/global/membership/`.
    project: Required. The name of the Fleet host project where this cluster
      will be registered. Project names are formatted as `projects/`.
  """
    membership = _messages.StringField(1)
    project = _messages.StringField(2)