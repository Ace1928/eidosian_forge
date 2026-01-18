from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1CloudBigtableSystemSpec(_messages.Message):
    """Specification that applies to all entries that are part of
  `CLOUD_BIGTABLE` system (user_specified_type)

  Fields:
    instanceDisplayName: Display name of the Instance. This is user specified
      and different from the resource name.
  """
    instanceDisplayName = _messages.StringField(1)