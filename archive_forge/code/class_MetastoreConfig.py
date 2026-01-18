from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetastoreConfig(_messages.Message):
    """Specifies a Metastore configuration.

  Fields:
    dataprocMetastoreService: Required. Resource name of an existing Dataproc
      Metastore service.Example:
      projects/[project_id]/locations/[dataproc_region]/services/[service-
      name]
  """
    dataprocMetastoreService = _messages.StringField(1)