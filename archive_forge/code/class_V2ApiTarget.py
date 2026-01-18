from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class V2ApiTarget(_messages.Message):
    """A restriction for a specific service and optionally one or multiple
  specific methods. Both fields are case insensitive.

  Fields:
    methods: Optional. List of one or more methods that can be called. If
      empty, all methods for the service are allowed. A wildcard (*) can be
      used as the last symbol. Valid examples:
      `google.cloud.translate.v2.TranslateService.GetSupportedLanguage`
      `TranslateText` `Get*` `translate.googleapis.com.Get*`
    service: The service for this restriction. It should be the canonical
      service name, for example: `translate.googleapis.com`. You can use
      [`gcloud services list`](/sdk/gcloud/reference/services/list) to get a
      list of services that are enabled in the project.
  """
    methods = _messages.StringField(1, repeated=True)
    service = _messages.StringField(2)