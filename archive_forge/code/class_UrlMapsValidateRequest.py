from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UrlMapsValidateRequest(_messages.Message):
    """A UrlMapsValidateRequest object.

  Enums:
    LoadBalancingSchemesValueListEntryValuesEnum:

  Fields:
    loadBalancingSchemes: Specifies the load balancer type(s) this validation
      request is for. Use EXTERNAL_MANAGED for global external Application
      Load Balancers and regional external Application Load Balancers. Use
      EXTERNAL for classic Application Load Balancers. Use INTERNAL_MANAGED
      for internal Application Load Balancers. For more information, refer to
      Choosing a load balancer. If unspecified, the load balancing scheme will
      be inferred from the backend service resources this URL map references.
      If that can not be inferred (for example, this URL map only references
      backend buckets, or this Url map is for rewrites and redirects only and
      doesn't reference any backends), EXTERNAL will be used as the default
      type. If specified, the scheme(s) must not conflict with the load
      balancing scheme of the backend service resources this Url map
      references.
    resource: Content of the UrlMap to be validated.
  """

    class LoadBalancingSchemesValueListEntryValuesEnum(_messages.Enum):
        """LoadBalancingSchemesValueListEntryValuesEnum enum type.

    Values:
      EXTERNAL: Signifies that this will be used for classic Application Load
        Balancers.
      EXTERNAL_MANAGED: Signifies that this will be used for Envoy-based
        global external Application Load Balancers.
      LOAD_BALANCING_SCHEME_UNSPECIFIED: If unspecified, the validation will
        try to infer the scheme from the backend service resources this Url
        map references. If the inference is not possible, EXTERNAL will be
        used as the default type.
    """
        EXTERNAL = 0
        EXTERNAL_MANAGED = 1
        LOAD_BALANCING_SCHEME_UNSPECIFIED = 2
    loadBalancingSchemes = _messages.EnumField('LoadBalancingSchemesValueListEntryValuesEnum', 1, repeated=True)
    resource = _messages.MessageField('UrlMap', 2)