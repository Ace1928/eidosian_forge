from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ImportRagFilesConfig(_messages.Message):
    """Config for importing RagFiles.

  Fields:
    gcsSource: Google Cloud Storage location. Supports importing individual
      files as well as entire Google Cloud Storage directories. Sample
      formats: - `gs://bucket_name/my_directory/object_name/my_file.txt` -
      `gs://bucket_name/my_directory`
    googleDriveSource: Google Drive location. Supports importing individual
      files as well as Google Drive folders.
    ragFileChunkingConfig: Specifies the size and overlap of chunks after
      importing RagFiles.
  """
    gcsSource = _messages.MessageField('GoogleCloudAiplatformV1beta1GcsSource', 1)
    googleDriveSource = _messages.MessageField('GoogleCloudAiplatformV1beta1GoogleDriveSource', 2)
    ragFileChunkingConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1RagFileChunkingConfig', 3)