from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ExportModelRequestOutputConfig(_messages.Message):
    """Output configuration for the Model export.

  Fields:
    artifactDestination: The Cloud Storage location where the Model artifact
      is to be written to. Under the directory given as the destination a new
      one with name "`model-export--`", where timestamp is in YYYY-MM-
      DDThh:mm:ss.sssZ ISO-8601 format, will be created. Inside, the Model and
      any of its supporting files will be written. This field should only be
      set when the `exportableContent` field of the
      [Model.supported_export_formats] object contains `ARTIFACT`.
    exportFormatId: The ID of the format in which the Model must be exported.
      Each Model lists the export formats it supports. If no value is provided
      here, then the first from the list of the Model's supported formats is
      used by default.
    imageDestination: The Google Container Registry or Artifact Registry uri
      where the Model container image will be copied to. This field should
      only be set when the `exportableContent` field of the
      [Model.supported_export_formats] object contains `IMAGE`.
  """
    artifactDestination = _messages.MessageField('GoogleCloudAiplatformV1GcsDestination', 1)
    exportFormatId = _messages.StringField(2)
    imageDestination = _messages.MessageField('GoogleCloudAiplatformV1ContainerRegistryDestination', 3)