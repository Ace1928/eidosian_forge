from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ModelExportFormat(_messages.Message):
    """Represents export format supported by the Model. All formats export to
  Google Cloud Storage.

  Enums:
    ExportableContentsValueListEntryValuesEnum:

  Fields:
    exportableContents: Output only. The content of this Model that may be
      exported.
    id: Output only. The ID of the export format. The possible format IDs are:
      * `tflite` Used for Android mobile devices. * `edgetpu-tflite` Used for
      [Edge TPU](https://cloud.google.com/edge-tpu/) devices. * `tf-saved-
      model` A tensorflow model in SavedModel format. * `tf-js` A
      [TensorFlow.js](https://www.tensorflow.org/js) model that can be used in
      the browser and in Node.js using JavaScript. * `core-ml` Used for iOS
      mobile devices. * `custom-trained` A Model that was uploaded or trained
      by custom code.
  """

    class ExportableContentsValueListEntryValuesEnum(_messages.Enum):
        """ExportableContentsValueListEntryValuesEnum enum type.

    Values:
      EXPORTABLE_CONTENT_UNSPECIFIED: Should not be used.
      ARTIFACT: Model artifact and any of its supported files. Will be
        exported to the location specified by the `artifactDestination` field
        of the ExportModelRequest.output_config object.
      IMAGE: The container image that is to be used when deploying this Model.
        Will be exported to the location specified by the `imageDestination`
        field of the ExportModelRequest.output_config object.
    """
        EXPORTABLE_CONTENT_UNSPECIFIED = 0
        ARTIFACT = 1
        IMAGE = 2
    exportableContents = _messages.EnumField('ExportableContentsValueListEntryValuesEnum', 1, repeated=True)
    id = _messages.StringField(2)