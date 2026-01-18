from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiService(_messages.Message):
    """`Service` is the root object of Google service configuration schema. It
  describes basic information about a service, such as the name and the title,
  and delegates other aspects to sub-sections. Each sub-section is either a
  proto message or a repeated proto message that configures a specific aspect,
  such as auth. See each proto message definition for details.  Example:
  type: google.api.Service     config_version: 3     name:
  calendar.googleapis.com     title: Google Calendar API     apis:     - name:
  google.calendar.v3.Calendar     authentication:       providers:       - id:
  google_calendar_auth         jwks_uri:
  https://www.googleapis.com/oauth2/v1/certs         issuer:
  https://securetoken.google.com       rules:       - selector: "*"
  requirements:           provider_id: google_calendar_auth

  Fields:
    apis: A list of API interfaces exported by this service. Only the `name`
      field of the google.protobuf.Api needs to be provided by the
      configuration author, as the remaining fields will be derived from the
      IDL during the normalization process. It is an error to specify an API
      interface here which cannot be resolved against the associated IDL
      files.
    authentication: Auth configuration.
    backend: API backend configuration.
    billing: Billing configuration.
    configVersion: The semantic version of the service configuration. The
      config version affects the interpretation of the service configuration.
      For example, certain features are enabled by default for certain config
      versions. The latest config version is `3`.
    context: Context configuration.
    control: Configuration for the service control plane.
    customError: Custom error configuration.
    documentation: Additional API documentation.
    endpoints: Configuration for network endpoints.  If this is empty, then an
      endpoint with the same name as the service is automatically generated to
      service all defined APIs.
    enums: A list of all enum types included in this API service.  Enums
      referenced directly or indirectly by the `apis` are automatically
      included.  Enums which are not referenced but shall be included should
      be listed here by name. Example:      enums:     - name:
      google.someapi.v1.SomeEnum
    http: HTTP configuration.
    id: A unique ID for a specific instance of this message, typically
      assigned by the client for tracking purpose. If empty, the server may
      choose to generate one instead. Must be no longer than 60 characters.
    logging: Logging configuration.
    logs: Defines the logs used by this service.
    metrics: Defines the metrics used by this service.
    monitoredResources: Defines the monitored resources used by this service.
      This is required by the Service.monitoring and Service.logging
      configurations.
    monitoring: Monitoring configuration.
    name: The service name, which is a DNS-like logical identifier for the
      service, such as `calendar.googleapis.com`. The service name typically
      goes through DNS verification to make sure the owner of the service also
      owns the DNS name.
    producerProjectId: The Google project that owns this service.
    quota: Quota configuration.
    sourceInfo: Output only. The source information for this configuration if
      available.
    systemParameters: System parameter configuration.
    systemTypes: A list of all proto message types included in this API
      service. It serves similar purpose as [google.api.Service.types], except
      that these types are not needed by user-defined APIs. Therefore, they
      will not show up in the generated discovery doc. This field should only
      be used to define system APIs in ESF.
    title: The product title for this service.
    types: A list of all proto message types included in this API service.
      Types referenced directly or indirectly by the `apis` are automatically
      included.  Messages which are not referenced but shall be included, such
      as types used by the `google.protobuf.Any` type, should be listed here
      by name. Example:      types:     - name: google.protobuf.Int32
    usage: Configuration controlling usage of this service.
  """
    apis = _messages.MessageField('Api', 1, repeated=True)
    authentication = _messages.MessageField('Authentication', 2)
    backend = _messages.MessageField('Backend', 3)
    billing = _messages.MessageField('Billing', 4)
    configVersion = _messages.IntegerField(5, variant=_messages.Variant.UINT32)
    context = _messages.MessageField('Context', 6)
    control = _messages.MessageField('Control', 7)
    customError = _messages.MessageField('CustomError', 8)
    documentation = _messages.MessageField('Documentation', 9)
    endpoints = _messages.MessageField('Endpoint', 10, repeated=True)
    enums = _messages.MessageField('Enum', 11, repeated=True)
    http = _messages.MessageField('Http', 12)
    id = _messages.StringField(13)
    logging = _messages.MessageField('Logging', 14)
    logs = _messages.MessageField('LogDescriptor', 15, repeated=True)
    metrics = _messages.MessageField('MetricDescriptor', 16, repeated=True)
    monitoredResources = _messages.MessageField('MonitoredResourceDescriptor', 17, repeated=True)
    monitoring = _messages.MessageField('Monitoring', 18)
    name = _messages.StringField(19)
    producerProjectId = _messages.StringField(20)
    quota = _messages.MessageField('Quota', 21)
    sourceInfo = _messages.MessageField('SourceInfo', 22)
    systemParameters = _messages.MessageField('SystemParameters', 23)
    systemTypes = _messages.MessageField('Type', 24, repeated=True)
    title = _messages.StringField(25)
    types = _messages.MessageField('Type', 26, repeated=True)
    usage = _messages.MessageField('Usage', 27)