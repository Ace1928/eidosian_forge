from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2CloudDlpInspection(_messages.Message):
    """Details about the Cloud Data Loss Prevention (Cloud DLP) [inspection
  job](https://cloud.google.com/dlp/docs/concepts-job-triggers) that produced
  the finding.

  Fields:
    fullScan: Whether Cloud DLP scanned the complete resource or a sampled
      subset.
    infoType: The type of information (or
      *[infoType](https://cloud.google.com/dlp/docs/infotypes-reference)*)
      found, for example, `EMAIL_ADDRESS` or `STREET_ADDRESS`.
    infoTypeCount: The number of times Cloud DLP found this infoType within
      this job and resource.
    inspectJob: Name of the inspection job, for example,
      `projects/123/locations/europe/dlpJobs/i-8383929`.
  """
    fullScan = _messages.BooleanField(1)
    infoType = _messages.StringField(2)
    infoTypeCount = _messages.IntegerField(3)
    inspectJob = _messages.StringField(4)