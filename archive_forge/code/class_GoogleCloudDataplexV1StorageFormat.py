from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1StorageFormat(_messages.Message):
    """Describes the format of the data within its storage location.

  Enums:
    CompressionFormatValueValuesEnum: Optional. The compression type
      associated with the stored data. If unspecified, the data is
      uncompressed.
    FormatValueValuesEnum: Output only. The data format associated with the
      stored data, which represents content type values. The value is inferred
      from mime type.

  Fields:
    compressionFormat: Optional. The compression type associated with the
      stored data. If unspecified, the data is uncompressed.
    csv: Optional. Additional information about CSV formatted data.
    format: Output only. The data format associated with the stored data,
      which represents content type values. The value is inferred from mime
      type.
    iceberg: Optional. Additional information about iceberg tables.
    json: Optional. Additional information about CSV formatted data.
    mimeType: Required. The mime type descriptor for the data. Must match the
      pattern {type}/{subtype}. Supported values: application/x-parquet
      application/x-avro application/x-orc application/x-tfrecord
      application/x-parquet+iceberg application/x-avro+iceberg
      application/x-orc+iceberg application/json application/{subtypes}
      text/csv text/ image/{image subtype} video/{video subtype} audio/{audio
      subtype}
  """

    class CompressionFormatValueValuesEnum(_messages.Enum):
        """Optional. The compression type associated with the stored data. If
    unspecified, the data is uncompressed.

    Values:
      COMPRESSION_FORMAT_UNSPECIFIED: CompressionFormat unspecified. Implies
        uncompressed data.
      GZIP: GZip compressed set of files.
      BZIP2: BZip2 compressed set of files.
    """
        COMPRESSION_FORMAT_UNSPECIFIED = 0
        GZIP = 1
        BZIP2 = 2

    class FormatValueValuesEnum(_messages.Enum):
        """Output only. The data format associated with the stored data, which
    represents content type values. The value is inferred from mime type.

    Values:
      FORMAT_UNSPECIFIED: Format unspecified.
      PARQUET: Parquet-formatted structured data.
      AVRO: Avro-formatted structured data.
      ORC: Orc-formatted structured data.
      CSV: Csv-formatted semi-structured data.
      JSON: Json-formatted semi-structured data.
      IMAGE: Image data formats (such as jpg and png).
      AUDIO: Audio data formats (such as mp3, and wav).
      VIDEO: Video data formats (such as mp4 and mpg).
      TEXT: Textual data formats (such as txt and xml).
      TFRECORD: TensorFlow record format.
      OTHER: Data that doesn't match a specific format.
      UNKNOWN: Data of an unknown format.
    """
        FORMAT_UNSPECIFIED = 0
        PARQUET = 1
        AVRO = 2
        ORC = 3
        CSV = 4
        JSON = 5
        IMAGE = 6
        AUDIO = 7
        VIDEO = 8
        TEXT = 9
        TFRECORD = 10
        OTHER = 11
        UNKNOWN = 12
    compressionFormat = _messages.EnumField('CompressionFormatValueValuesEnum', 1)
    csv = _messages.MessageField('GoogleCloudDataplexV1StorageFormatCsvOptions', 2)
    format = _messages.EnumField('FormatValueValuesEnum', 3)
    iceberg = _messages.MessageField('GoogleCloudDataplexV1StorageFormatIcebergOptions', 4)
    json = _messages.MessageField('GoogleCloudDataplexV1StorageFormatJsonOptions', 5)
    mimeType = _messages.StringField(6)