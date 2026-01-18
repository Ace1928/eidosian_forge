from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HttpRoute(_messages.Message):
    """HttpRoute is the resource defining how HTTP traffic should be routed by
  a Mesh or Gateway resource.

  Messages:
    LabelsValue: Optional. Set of label tags associated with the HttpRoute
      resource.

  Fields:
    createTime: Output only. The timestamp when the resource was created.
    description: Optional. A free-text description of the resource. Max length
      1024 characters.
    gateways: Optional. Gateways defines a list of gateways this HttpRoute is
      attached to, as one of the routing rules to route the requests served by
      the gateway. Each gateway reference should match the pattern:
      `projects/*/locations/global/gateways/`
    hostnames: Required. Hostnames define a set of hosts that should match
      against the HTTP host header to select a HttpRoute to process the
      request. Hostname is the fully qualified domain name of a network host,
      as defined by RFC 1123 with the exception that: - IPs are not allowed. -
      A hostname may be prefixed with a wildcard label (`*.`). The wildcard
      label must appear by itself as the first label. Hostname can be
      "precise" which is a domain name without the terminating dot of a
      network host (e.g. `foo.example.com`) or "wildcard", which is a domain
      name prefixed with a single wildcard label (e.g. `*.example.com`). Note
      that as per RFC1035 and RFC1123, a label must consist of lower case
      alphanumeric characters or '-', and must start and end with an
      alphanumeric character. No other punctuation is allowed. The routes
      associated with a Mesh or Gateways must have unique hostnames. If you
      attempt to attach multiple routes with conflicting hostnames, the
      configuration will be rejected. For example, while it is acceptable for
      routes for the hostnames `*.foo.bar.com` and `*.bar.com` to be
      associated with the same Mesh (or Gateways under the same scope), it is
      not possible to associate two routes both with `*.bar.com` or both with
      `bar.com`.
    labels: Optional. Set of label tags associated with the HttpRoute
      resource.
    meshes: Optional. Meshes defines a list of meshes this HttpRoute is
      attached to, as one of the routing rules to route the requests served by
      the mesh. Each mesh reference should match the pattern:
      `projects/*/locations/global/meshes/` The attached Mesh should be of a
      type SIDECAR
    name: Required. Name of the HttpRoute resource. It matches pattern
      `projects/*/locations/global/httpRoutes/http_route_name>`.
    rules: Required. Rules that define how traffic is routed and handled.
      Rules will be matched sequentially based on the RouteMatch specified for
      the rule.
    selfLink: Output only. Server-defined URL of this resource
    updateTime: Output only. The timestamp when the resource was updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Set of label tags associated with the HttpRoute resource.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    gateways = _messages.StringField(3, repeated=True)
    hostnames = _messages.StringField(4, repeated=True)
    labels = _messages.MessageField('LabelsValue', 5)
    meshes = _messages.StringField(6, repeated=True)
    name = _messages.StringField(7)
    rules = _messages.MessageField('HttpRouteRouteRule', 8, repeated=True)
    selfLink = _messages.StringField(9)
    updateTime = _messages.StringField(10)