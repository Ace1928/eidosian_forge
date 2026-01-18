from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2CloudStorageOptions(_messages.Message):
    """Options defining a file or a set of files within a Cloud Storage bucket.

  Enums:
    FileTypesValueListEntryValuesEnum:
    SampleMethodValueValuesEnum: How to sample the data.

  Fields:
    bytesLimitPerFile: Max number of bytes to scan from a file. If a scanned
      file's size is bigger than this value then the rest of the bytes are
      omitted. Only one of `bytes_limit_per_file` and
      `bytes_limit_per_file_percent` can be specified. This field can't be set
      if de-identification is requested. For certain file types, setting this
      field has no effect. For more information, see [Limits on bytes scanned
      per file](https://cloud.google.com/sensitive-data-
      protection/docs/supported-file-types#max-byte-size-per-file).
    bytesLimitPerFilePercent: Max percentage of bytes to scan from a file. The
      rest are omitted. The number of bytes scanned is rounded down. Must be
      between 0 and 100, inclusively. Both 0 and 100 means no limit. Defaults
      to 0. Only one of bytes_limit_per_file and bytes_limit_per_file_percent
      can be specified. This field can't be set if de-identification is
      requested. For certain file types, setting this field has no effect. For
      more information, see [Limits on bytes scanned per
      file](https://cloud.google.com/sensitive-data-protection/docs/supported-
      file-types#max-byte-size-per-file).
    fileSet: The set of one or more files to scan.
    fileTypes: List of file type groups to include in the scan. If empty, all
      files are scanned and available data format processors are applied. In
      addition, the binary content of the selected files is always scanned as
      well. Images are scanned only as binary if the specified region does not
      support image inspection and no file_types were specified. Image
      inspection is restricted to 'global', 'us', 'asia', and 'europe'.
    filesLimitPercent: Limits the number of files to scan to this percentage
      of the input FileSet. Number of files scanned is rounded down. Must be
      between 0 and 100, inclusively. Both 0 and 100 means no limit. Defaults
      to 0.
    sampleMethod: How to sample the data.
  """

    class FileTypesValueListEntryValuesEnum(_messages.Enum):
        """FileTypesValueListEntryValuesEnum enum type.

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

    class SampleMethodValueValuesEnum(_messages.Enum):
        """How to sample the data.

    Values:
      SAMPLE_METHOD_UNSPECIFIED: No sampling.
      TOP: Scan from the top (default).
      RANDOM_START: For each file larger than bytes_limit_per_file, randomly
        pick the offset to start scanning. The scanned bytes are contiguous.
    """
        SAMPLE_METHOD_UNSPECIFIED = 0
        TOP = 1
        RANDOM_START = 2
    bytesLimitPerFile = _messages.IntegerField(1)
    bytesLimitPerFilePercent = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    fileSet = _messages.MessageField('GooglePrivacyDlpV2FileSet', 3)
    fileTypes = _messages.EnumField('FileTypesValueListEntryValuesEnum', 4, repeated=True)
    filesLimitPercent = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    sampleMethod = _messages.EnumField('SampleMethodValueValuesEnum', 6)