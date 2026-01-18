from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsGetDeployedIngressConfigRequest(_messages.Message):
    """A ApigeeOrganizationsGetDeployedIngressConfigRequest object.

  Enums:
    ViewValueValuesEnum: When set to FULL, additional details about the
      specific deployments receiving traffic will be included in the
      IngressConfig response's RoutingRules.

  Fields:
    name: Required. Name of the deployed configuration for the organization in
      the following format: 'organizations/{org}/deployedIngressConfig'.
    view: When set to FULL, additional details about the specific deployments
      receiving traffic will be included in the IngressConfig response's
      RoutingRules.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """When set to FULL, additional details about the specific deployments
    receiving traffic will be included in the IngressConfig response's
    RoutingRules.

    Values:
      INGRESS_CONFIG_VIEW_UNSPECIFIED: The default/unset value. The API will
        default to the BASIC view.
      BASIC: Include all ingress config data necessary for the runtime to
        configure ingress, but no more. Routing rules will include only
        basepath and destination environment. This the default value.
      FULL: Include all ingress config data, including internal debug info for
        each routing rule such as the proxy claiming a particular basepath and
        when the routing rule first appeared in the env group.
    """
        INGRESS_CONFIG_VIEW_UNSPECIFIED = 0
        BASIC = 1
        FULL = 2
    name = _messages.StringField(1, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 2)