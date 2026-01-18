from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class DemoteMasterConfiguration(_messages.Message):
    """Read-replica configuration for connecting to the on-premises primary
  instance.

  Fields:
    kind: This is always `sql#demoteMasterConfiguration`.
    mysqlReplicaConfiguration: MySQL specific configuration when replicating
      from a MySQL on-premises primary instance. Replication configuration
      information such as the username, password, certificates, and keys are
      not stored in the instance metadata. The configuration information is
      used only to set up the replication connection and is stored by MySQL in
      a file named `master.info` in the data directory.
  """
    kind = _messages.StringField(1)
    mysqlReplicaConfiguration = _messages.MessageField('DemoteMasterMySqlReplicaConfiguration', 2)