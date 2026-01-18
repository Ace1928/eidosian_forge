from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InputConfig(_messages.Message):
    """Input configuration for BatchTranslateText request.

  Fields:
    gcsSource: Required. Google Cloud Storage location for the source input.
      This can be a single file (for example, `gs://translation-
      test/input.tsv`) or a wildcard (for example, `gs://translation-test/*`).
      If a file extension is `.tsv`, it can contain either one or two columns.
      The first column (optional) is the id of the text request. If the first
      column is missing, we use the row number (0-based) from the input file
      as the ID in the output file. The second column is the actual text to be
      translated. We recommend each row be <= 10K Unicode codepoints,
      otherwise an error might be returned. Note that the input tsv must be
      RFC 4180 compliant. You could use https://github.com/Clever/csvlint to
      check potential formatting errors in your tsv file. csvlint
      --delimiter='\\t' your_input_file.tsv The other supported file extensions
      are `.txt` or `.html`, which is treated as a single large chunk of text.
    mimeType: Optional. Can be "text/plain" or "text/html". For `.tsv`,
      "text/html" is used if mime_type is missing. For `.html`, this field
      must be "text/html" or empty. For `.txt`, this field must be
      "text/plain" or empty.
  """
    gcsSource = _messages.MessageField('GcsSource', 1)
    mimeType = _messages.StringField(2)