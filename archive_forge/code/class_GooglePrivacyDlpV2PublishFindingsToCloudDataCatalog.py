from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2PublishFindingsToCloudDataCatalog(_messages.Message):
    """Publish findings of a DlpJob to Data Catalog. In Data Catalog, tag
  templates are applied to the resource that Cloud DLP scanned. Data Catalog
  tag templates are stored in the same project and region where the BigQuery
  table exists. For Cloud DLP to create and apply the tag template, the Cloud
  DLP service agent must have the `roles/datacatalog.tagTemplateOwner`
  permission on the project. The tag template contains fields summarizing the
  results of the DlpJob. Any field values previously written by another DlpJob
  are deleted. InfoType naming patterns are strictly enforced when using this
  feature. Findings are persisted in Data Catalog storage and are governed by
  service-specific policies for Data Catalog. For more information, see
  [Service Specific Terms](https://cloud.google.com/terms/service-terms). Only
  a single instance of this action can be specified. This action is allowed
  only if all resources being scanned are BigQuery tables. Compatible with:
  Inspect
  """