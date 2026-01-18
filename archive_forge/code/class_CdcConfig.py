from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CdcConfig(_messages.Message):
    """Configuration information to start the Change Data Capture (CDC) streams
  from customer database to backend database of Dataproc Metastore.

  Fields:
    bucket: Optional. The bucket to write the intermediate stream event data
      in. The bucket name must be without any prefix like "gs://". See the
      bucket naming requirements
      (https://cloud.google.com/storage/docs/buckets#naming). This field is
      optional. If not set, the Artifacts Cloud Storage bucket will be used.
    password: Required. Input only. The password for the user that Datastream
      service should use for the MySQL connection. This field is not returned
      on request.
    reverseProxySubnet: Required. The URL of the subnetwork resource to create
      the VM instance hosting the reverse proxy in. More context in
      https://cloud.google.com/datastream/docs/private-connectivity#reverse-
      csql-proxy The subnetwork should reside in the network provided in the
      request that Datastream will peer to and should be in the same region as
      Datastream, in the following format.
      projects/{project_id}/regions/{region_id}/subnetworks/{subnetwork_id}
    rootPath: Optional. The root path inside the Cloud Storage bucket. The
      stream event data will be written to this path. The default value is
      /migration.
    subnetIpRange: Required. A /29 CIDR IP range for peering with datastream.
    username: Required. The username that the Datastream service should use
      for the MySQL connection.
    vpcNetwork: Required. Fully qualified name of the Cloud SQL instance's VPC
      network or the shared VPC network that Datastream will peer to, in the
      following format:
      projects/{project_id}/locations/global/networks/{network_id}. More
      context in https://cloud.google.com/datastream/docs/network-
      connectivity-options#privateconnectivity
  """
    bucket = _messages.StringField(1)
    password = _messages.StringField(2)
    reverseProxySubnet = _messages.StringField(3)
    rootPath = _messages.StringField(4)
    subnetIpRange = _messages.StringField(5)
    username = _messages.StringField(6)
    vpcNetwork = _messages.StringField(7)