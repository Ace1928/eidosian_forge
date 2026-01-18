from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta1OutputConfig(_messages.Message):
    """The desired output location and metadata.

  Fields:
    gcsDestination: The Google Cloud Storage location to write the output to.
    pagesPerShard: The max number of pages to include into each output
      Document shard JSON on Google Cloud Storage. The valid range is [1,
      100]. If not specified, the default value is 20. For example, for one
      pdf file with 100 pages, 100 parsed pages will be produced. If
      `pages_per_shard` = 20, then 5 Document shard JSON files each containing
      20 parsed pages will be written under the prefix
      OutputConfig.gcs_destination.uri and suffix pages-x-to-y.json where x
      and y are 1-indexed page numbers. Example GCS outputs with 157 pages and
      pages_per_shard = 50: pages-001-to-050.json pages-051-to-100.json
      pages-101-to-150.json pages-151-to-157.json
  """
    gcsDestination = _messages.MessageField('GoogleCloudDocumentaiV1beta1GcsDestination', 1)
    pagesPerShard = _messages.IntegerField(2, variant=_messages.Variant.INT32)