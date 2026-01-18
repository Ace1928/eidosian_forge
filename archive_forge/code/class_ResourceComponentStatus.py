from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourceComponentStatus(_messages.Message):
    """Status for a component of a resource.

  Enums:
    StateValueValuesEnum: The state of the resource component.

  Fields:
    consoleLink: Pantheon link for the resource. This does not exist for every
      resource that makes up the SAF resource.
    diverged: Indicates that this resource component has been altered and may
      not match the expected state.
    name: The name the resource component. Usually it's the name of the GCP
      resource, which was used inside the Terraform Resource block that
      defines it. (e.g. cri-domain-cert)
    reason: The reason why this resource component to be in its state.
    selfLink: Fully qualified URL to the object represented by this resource
      component.
    state: The state of the resource component.
    type: The Terraform Resource Type of the GCP resource (e.g.
      "google_compute_managed_ssl_certificate").
  """

    class StateValueValuesEnum(_messages.Enum):
        """The state of the resource component.

    Values:
      STATE_UNSPECIFIED: The status of this component is unspecified.
      DEPLOYED: The component has been deployed.
      MISSING: The component is missing.
      PROVISIONING: The component has been deployed and is provisioning.
      ACTIVE: The component has been deployed and is working as intended. This
        is intended for resources that have a health indicator.
      FAILED: The component has failed and the full error message will be
        populated in the resource.
    """
        STATE_UNSPECIFIED = 0
        DEPLOYED = 1
        MISSING = 2
        PROVISIONING = 3
        ACTIVE = 4
        FAILED = 5
    consoleLink = _messages.StringField(1)
    diverged = _messages.BooleanField(2)
    name = _messages.StringField(3)
    reason = _messages.StringField(4)
    selfLink = _messages.StringField(5)
    state = _messages.EnumField('StateValueValuesEnum', 6)
    type = _messages.StringField(7)