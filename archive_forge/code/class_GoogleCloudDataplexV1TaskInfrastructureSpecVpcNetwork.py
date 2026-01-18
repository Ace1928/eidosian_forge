from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1TaskInfrastructureSpecVpcNetwork(_messages.Message):
    """Cloud VPC Network used to run the infrastructure.

  Fields:
    network: Optional. The Cloud VPC network in which the job is run. By
      default, the Cloud VPC network named Default within the project is used.
    networkTags: Optional. List of network tags to apply to the job.
    subNetwork: Optional. The Cloud VPC sub-network in which the job is run.
  """
    network = _messages.StringField(1)
    networkTags = _messages.StringField(2, repeated=True)
    subNetwork = _messages.StringField(3)