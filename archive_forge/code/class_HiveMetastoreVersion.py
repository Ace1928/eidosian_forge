from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HiveMetastoreVersion(_messages.Message):
    """A specification of a supported version of the Hive Metastore software.

  Fields:
    isDefault: Whether version will be chosen by the server if a metastore
      service is created with a HiveMetastoreConfig that omits the version.
    version: The semantic version of the Hive Metastore software.
  """
    isDefault = _messages.BooleanField(1)
    version = _messages.StringField(2)