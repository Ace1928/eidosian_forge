from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagedService(_messages.Message):
    """The full representation of an API Service that is managed by the
  `ServiceManager` API.  Includes both the service configuration, as well as
  other control plane deployment related information.

  Fields:
    configSource: User-supplied source configuration for the service. This is
      distinct from the generated configuration provided in
      `google.api.Service`. This is NOT populated on GetService calls at the
      moment. NOTE: Any upsert operation that contains both a service_config
      and a config_source is considered invalid and will result in an error
      being returned.
    generation: A server-assigned monotonically increasing number that changes
      whenever a mutation is made to the `ManagedService` or any of its
      components via the `ServiceManager` API.
    operations: Read-only view of pending operations affecting this resource,
      if requested.
    producerProjectId: ID of the project that produces and owns this service.
    projectSettings: Read-only view of settings for a particular consumer
      project, if requested.
    serviceConfig: The service's generated configuration.
    serviceName: The name of the service.  See the `ServiceManager` overview
      for naming requirements.  This name must match `google.api.Service.name`
      in the `service_config` field.
  """
    configSource = _messages.MessageField('ConfigSource', 1)
    generation = _messages.IntegerField(2)
    operations = _messages.MessageField('Operation', 3, repeated=True)
    producerProjectId = _messages.StringField(4)
    projectSettings = _messages.MessageField('ProjectSettings', 5)
    serviceConfig = _messages.MessageField('Service', 6)
    serviceName = _messages.StringField(7)