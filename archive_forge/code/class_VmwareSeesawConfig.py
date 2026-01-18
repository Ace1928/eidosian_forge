from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareSeesawConfig(_messages.Message):
    """VmwareSeesawConfig represents configuration parameters for an already
  existing Seesaw load balancer. IMPORTANT: Please note that the Anthos On-
  Prem API will not generate or update Seesaw configurations it can only bind
  a pre-existing configuration to a new user cluster. IMPORTANT: When
  attempting to create a user cluster with a pre-existing Seesaw load balancer
  you will need to follow some preparation steps before calling the
  'CreateVmwareCluster' API method. First you will need to create the user
  cluster's namespace via kubectl. The namespace will need to use the
  following naming convention : -gke-onprem-mgmt or -gke-onprem-mgmt depending
  on whether you used the 'VmwareCluster.local_name' to disambiguate
  collisions; for more context see the documentation of
  'VmwareCluster.local_name'. Once the namespace is created you will need to
  create a secret resource via kubectl. This secret will contain copies of
  your Seesaw credentials. The Secret must be called 'user-cluster-creds' and
  contain Seesaw's SSH and Cert credentials. The credentials must be keyed
  with the following names: 'seesaw-ssh-private-key', 'seesaw-ssh-public-key',
  'seesaw-ssh-ca-key', 'seesaw-ssh-ca-cert'.

  Fields:
    enableHa: Enable two load balancer VMs to achieve a highly-available
      Seesaw load balancer.
    group: Required. In general the following format should be used for the
      Seesaw group name: seesaw-for-[cluster_name].
    ipBlocks: Required. The IP Blocks to be used by the Seesaw load balancer
    masterIp: Required. MasterIP is the IP announced by the master of Seesaw
      group.
    stackdriverName: Name to be used by Stackdriver.
    vms: Names of the VMs created for this Seesaw group.
  """
    enableHa = _messages.BooleanField(1)
    group = _messages.StringField(2)
    ipBlocks = _messages.MessageField('VmwareIpBlock', 3, repeated=True)
    masterIp = _messages.StringField(4)
    stackdriverName = _messages.StringField(5)
    vms = _messages.StringField(6, repeated=True)