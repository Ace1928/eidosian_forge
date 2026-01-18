from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ExportDataResponse(_messages.Message):
    """Response message for DatasetService.ExportData.

  Fields:
    exportedFiles: All of the files that are exported in this export
      operation. For custom code training export, only three (training,
      validation and test) Cloud Storage paths in wildcard format are
      populated (for example, gs://.../training-*).
  """
    exportedFiles = _messages.StringField(1, repeated=True)