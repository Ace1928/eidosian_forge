from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1StorageProperties(_messages.Message):
    """Details the properties of the underlying storage.

  Fields:
    filePattern: Patterns to identify a set of files for this fileset.
      Examples of a valid `file_pattern`: * `gs://bucket_name/dir/*`: matches
      all files in the `bucket_name/dir` directory *
      `gs://bucket_name/dir/**`: matches all files in the `bucket_name/dir`
      and all subdirectories recursively * `gs://bucket_name/file*`: matches
      files prefixed by `file` in `bucket_name` * `gs://bucket_name/??.txt`:
      matches files with two characters followed by `.txt` in `bucket_name` *
      `gs://bucket_name/[aeiou].txt`: matches files that contain a single
      vowel character followed by `.txt` in `bucket_name` *
      `gs://bucket_name/[a-m].txt`: matches files that contain `a`, `b`, ...
      or `m` followed by `.txt` in `bucket_name` * `gs://bucket_name/a/*/b`:
      matches all files in `bucket_name` that match the `a/*/b` pattern, such
      as `a/c/b`, `a/d/b` * `gs://another_bucket/a.txt`: matches
      `gs://another_bucket/a.txt`
    fileType: File type in MIME format, for example, `text/plain`.
  """
    filePattern = _messages.StringField(1, repeated=True)
    fileType = _messages.StringField(2)