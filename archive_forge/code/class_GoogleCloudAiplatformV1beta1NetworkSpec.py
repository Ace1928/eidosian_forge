from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1NetworkSpec(_messages.Message):
    """Network spec.

  Fields:
    enableInternetAccess: Whether to enable public internet access. Default
      false.
    network: The full name of the Google Compute Engine
      [network](https://cloud.google.com//compute/docs/networks-and-
      firewalls#networks)
    subnetwork: The name of the subnet that this instance is in. Format: `proj
      ects/{project_id_or_number}/regions/{region}/subnetworks/{subnetwork_id}
      `
  """
    enableInternetAccess = _messages.BooleanField(1)
    network = _messages.StringField(2)
    subnetwork = _messages.StringField(3)