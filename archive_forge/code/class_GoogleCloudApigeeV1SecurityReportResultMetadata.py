from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1SecurityReportResultMetadata(_messages.Message):
    """Contains informations about the security report results.

  Fields:
    expires: Output only. Expire_time is set to 7 days after report creation.
      Query result will be unaccessable after this time. Example:
      "2021-05-04T13:38:52-07:00"
    self: Self link of the query results. Example: `/organizations/myorg/envir
      onments/myenv/securityReports/9cfc0d85-0f30-46d6-ae6f-
      318d0cb961bd/result` or following format if query is running at host
      level: `/organizations/myorg/hostSecurityReports/9cfc0d85-0f30-46d6-
      ae6f-318d0cb961bd/result`
  """
    expires = _messages.StringField(1)
    self = _messages.StringField(2)