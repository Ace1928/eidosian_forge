from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2Deidentify(_messages.Message):
    """Create a de-identified copy of the requested table or files. A
  TransformationDetail will be created for each transformation. If any rows in
  BigQuery are skipped during de-identification (transformation errors or row
  size exceeds BigQuery insert API limits) they are placed in the failure
  output table. If the original row exceeds the BigQuery insert API limit it
  will be truncated when written to the failure output table. The failure
  output table can be set in the
  action.deidentify.output.big_query_output.deidentified_failure_output_table
  field, if no table is set, a table will be automatically created in the same
  project and dataset as the original table. Compatible with: Inspect

  Enums:
    FileTypesToTransformValueListEntryValuesEnum:

  Fields:
    cloudStorageOutput: Required. User settable Cloud Storage bucket and
      folders to store de-identified files. This field must be set for cloud
      storage deidentification. The output Cloud Storage bucket must be
      different from the input bucket. De-identified files will overwrite
      files in the output path. Form of: gs://bucket/folder/ or gs://bucket
    fileTypesToTransform: List of user-specified file type groups to
      transform. If specified, only the files with these filetypes will be
      transformed. If empty, all supported files will be transformed.
      Supported types may be automatically added over time. If a file type is
      set in this field that isn't supported by the Deidentify action then the
      job will fail and will not be successfully created/started. Currently
      the only filetypes supported are: IMAGES, TEXT_FILES, CSV, TSV.
    transformationConfig: User specified deidentify templates and configs for
      structured, unstructured, and image files.
    transformationDetailsStorageConfig: Config for storing transformation
      details. This is separate from the de-identified content, and contains
      metadata about the successful transformations and/or failures that
      occurred while de-identifying. This needs to be set in order for users
      to access information about the status of each transformation (see
      TransformationDetails message for more information about what is noted).
  """

    class FileTypesToTransformValueListEntryValuesEnum(_messages.Enum):
        """FileTypesToTransformValueListEntryValuesEnum enum type.

    Values:
      FILE_TYPE_UNSPECIFIED: Includes all files.
      BINARY_FILE: Includes all file extensions not covered by another entry.
        Binary scanning attempts to convert the content of the file to utf_8
        to scan the file. If you wish to avoid this fall back, specify one or
        more of the other file types in your storage scan.
      TEXT_FILE: Included file extensions: asc,asp, aspx, brf, c, cc,cfm, cgi,
        cpp, csv, cxx, c++, cs, css, dart, dat, dot, eml,, epbub, ged, go, h,
        hh, hpp, hxx, h++, hs, html, htm, mkd, markdown, m, ml, mli, perl, pl,
        plist, pm, php, phtml, pht, properties, py, pyw, rb, rbw, rs, rss, rc,
        scala, sh, sql, swift, tex, shtml, shtm, xhtml, lhs, ics, ini, java,
        js, json, jsonl, kix, kml, ocaml, md, txt, text, tsv, vb, vcard, vcs,
        wml, xcodeproj, xml, xsl, xsd, yml, yaml.
      IMAGE: Included file extensions: bmp, gif, jpg, jpeg, jpe, png. Setting
        bytes_limit_per_file or bytes_limit_per_file_percent has no effect on
        image files. Image inspection is restricted to the `global`, `us`,
        `asia`, and `europe` regions.
      WORD: Microsoft Word files larger than 30 MB will be scanned as binary
        files. Included file extensions: docx, dotx, docm, dotm. Setting
        `bytes_limit_per_file` or `bytes_limit_per_file_percent` has no effect
        on Word files.
      PDF: PDF files larger than 30 MB will be scanned as binary files.
        Included file extensions: pdf. Setting `bytes_limit_per_file` or
        `bytes_limit_per_file_percent` has no effect on PDF files.
      AVRO: Included file extensions: avro
      CSV: Included file extensions: csv
      TSV: Included file extensions: tsv
      POWERPOINT: Microsoft PowerPoint files larger than 30 MB will be scanned
        as binary files. Included file extensions: pptx, pptm, potx, potm,
        pot. Setting `bytes_limit_per_file` or `bytes_limit_per_file_percent`
        has no effect on PowerPoint files.
      EXCEL: Microsoft Excel files larger than 30 MB will be scanned as binary
        files. Included file extensions: xlsx, xlsm, xltx, xltm. Setting
        `bytes_limit_per_file` or `bytes_limit_per_file_percent` has no effect
        on Excel files.
    """
        FILE_TYPE_UNSPECIFIED = 0
        BINARY_FILE = 1
        TEXT_FILE = 2
        IMAGE = 3
        WORD = 4
        PDF = 5
        AVRO = 6
        CSV = 7
        TSV = 8
        POWERPOINT = 9
        EXCEL = 10
    cloudStorageOutput = _messages.StringField(1)
    fileTypesToTransform = _messages.EnumField('FileTypesToTransformValueListEntryValuesEnum', 2, repeated=True)
    transformationConfig = _messages.MessageField('GooglePrivacyDlpV2TransformationConfig', 3)
    transformationDetailsStorageConfig = _messages.MessageField('GooglePrivacyDlpV2TransformationDetailsStorageConfig', 4)