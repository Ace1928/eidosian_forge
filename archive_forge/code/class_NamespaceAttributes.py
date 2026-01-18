from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NamespaceAttributes(_messages.Message):
    """Attributes associated with Namespace.

  Fields:
    cloudDnsManagedZones: Output only. List of Cloud DNS ManagedZones that
      this namespace is associated with.
    managedRegistration: Output only. Indicates whether a GCP product or
      service manages this resource. When a resource is fully managed by
      another GCP product or system the information in Service Directory is
      read-only. The source of truth is the relevant GCP product or system
      which is managing the resource. The Service Directory resource will be
      updated or deleted as appropriate to reflect the state of the underlying
      `origin_resource`. Note: The `origin_resource` can be found in the
      endpoint(s) associated with service(s) associated with this namespace.
  """
    cloudDnsManagedZones = _messages.StringField(1, repeated=True)
    managedRegistration = _messages.BooleanField(2)