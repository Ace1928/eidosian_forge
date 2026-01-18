from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2Container(_messages.Message):
    """Represents a container that may contain DLP findings. Examples of a
  container include a file, table, or database record.

  Fields:
    fullPath: A string representation of the full container name. Examples: -
      BigQuery: 'Project:DataSetId.TableId' - Cloud Storage:
      'gs://Bucket/folders/filename.txt'
    projectId: Project where the finding was found. Can be different from the
      project that owns the finding.
    relativePath: The rest of the path after the root. Examples: - For
      BigQuery table `project_id:dataset_id.table_id`, the relative path is
      `table_id` - For Cloud Storage file `gs://bucket/folder/filename.txt`,
      the relative path is `folder/filename.txt`
    rootPath: The root of the container. Examples: - For BigQuery table
      `project_id:dataset_id.table_id`, the root is `dataset_id` - For Cloud
      Storage file `gs://bucket/folder/filename.txt`, the root is
      `gs://bucket`
    type: Container type, for example BigQuery or Cloud Storage.
    updateTime: Findings container modification timestamp, if applicable. For
      Cloud Storage, this field contains the last file modification timestamp.
      For a BigQuery table, this field contains the last_modified_time
      property. For Datastore, this field isn't populated.
    version: Findings container version, if available ("generation" for Cloud
      Storage).
  """
    fullPath = _messages.StringField(1)
    projectId = _messages.StringField(2)
    relativePath = _messages.StringField(3)
    rootPath = _messages.StringField(4)
    type = _messages.StringField(5)
    updateTime = _messages.StringField(6)
    version = _messages.StringField(7)