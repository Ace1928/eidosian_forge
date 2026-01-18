from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudSQLConnectionConfig(_messages.Message):
    """Configuration information to establish customer database connection
  before the cutover phase of migration

  Fields:
    hiveDatabaseName: Required. The hive database name.
    instanceConnectionName: Required. Cloud SQL database connection name
      (project_id:region:instance_name)
    ipAddress: Required. The private IP address of the Cloud SQL instance.
    natSubnet: Required. The relative resource name of the subnetwork to be
      used for Private Service Connect. Note that this cannot be a regular
      subnet and is used only for NAT.
      (https://cloud.google.com/vpc/docs/about-vpc-hosted-services#psc-
      subnets) This subnet is used to publish the SOCKS5 proxy service. The
      subnet size must be at least /29 and it should reside in a network
      through which the Cloud SQL instance is accessible. The resource name
      should be in the format,
      projects/{project_id}/regions/{region_id}/subnetworks/{subnetwork_id}
    password: Required. Input only. The password for the user that Dataproc
      Metastore service will be using to connect to the database. This field
      is not returned on request.
    port: Required. The network port of the database.
    proxySubnet: Required. The relative resource name of the subnetwork to
      deploy the SOCKS5 proxy service in. The subnetwork should reside in a
      network through which the Cloud SQL instance is accessible. The resource
      name should be in the format,
      projects/{project_id}/regions/{region_id}/subnetworks/{subnetwork_id}
    username: Required. The username that Dataproc Metastore service will use
      to connect to the database.
  """
    hiveDatabaseName = _messages.StringField(1)
    instanceConnectionName = _messages.StringField(2)
    ipAddress = _messages.StringField(3)
    natSubnet = _messages.StringField(4)
    password = _messages.StringField(5)
    port = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    proxySubnet = _messages.StringField(7)
    username = _messages.StringField(8)