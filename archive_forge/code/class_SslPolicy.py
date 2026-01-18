from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SslPolicy(_messages.Message):
    """Represents an SSL Policy resource. Use SSL policies to control SSL
  features, such as versions and cipher suites, that are offered by
  Application Load Balancers and proxy Network Load Balancers. For more
  information, read SSL policies overview.

  Enums:
    MinTlsVersionValueValuesEnum: The minimum version of SSL protocol that can
      be used by the clients to establish a connection with the load balancer.
      This can be one of TLS_1_0, TLS_1_1, TLS_1_2.
    ProfileValueValuesEnum: Profile specifies the set of SSL features that can
      be used by the load balancer when negotiating SSL with clients. This can
      be one of COMPATIBLE, MODERN, RESTRICTED, or CUSTOM. If using CUSTOM,
      the set of SSL features to enable must be specified in the
      customFeatures field.

  Messages:
    WarningsValueListEntry: A WarningsValueListEntry object.

  Fields:
    creationTimestamp: [Output Only] Creation timestamp in RFC3339 text
      format.
    customFeatures: A list of features enabled when the selected profile is
      CUSTOM. The method returns the set of features that can be specified in
      this list. This field must be empty if the profile is not CUSTOM.
    description: An optional description of this resource. Provide this
      property when you create the resource.
    enabledFeatures: [Output Only] The list of features enabled in the SSL
      policy.
    fingerprint: Fingerprint of this resource. A hash of the contents stored
      in this object. This field is used in optimistic locking. This field
      will be ignored when inserting a SslPolicy. An up-to-date fingerprint
      must be provided in order to update the SslPolicy, otherwise the request
      will fail with error 412 conditionNotMet. To see the latest fingerprint,
      make a get() request to retrieve an SslPolicy.
    id: [Output Only] The unique identifier for the resource. This identifier
      is defined by the server.
    kind: [Output only] Type of the resource. Always compute#sslPolicyfor SSL
      policies.
    minTlsVersion: The minimum version of SSL protocol that can be used by the
      clients to establish a connection with the load balancer. This can be
      one of TLS_1_0, TLS_1_1, TLS_1_2.
    name: Name of the resource. The name must be 1-63 characters long, and
      comply with RFC1035. Specifically, the name must be 1-63 characters long
      and match the regular expression `[a-z]([-a-z0-9]*[a-z0-9])?` which
      means the first character must be a lowercase letter, and all following
      characters must be a dash, lowercase letter, or digit, except the last
      character, which cannot be a dash.
    profile: Profile specifies the set of SSL features that can be used by the
      load balancer when negotiating SSL with clients. This can be one of
      COMPATIBLE, MODERN, RESTRICTED, or CUSTOM. If using CUSTOM, the set of
      SSL features to enable must be specified in the customFeatures field.
    region: [Output Only] URL of the region where the regional SSL policy
      resides. This field is not applicable to global SSL policies.
    selfLink: [Output Only] Server-defined URL for the resource.
    warnings: [Output Only] If potential misconfigurations are detected for
      this SSL policy, this field will be populated with warning messages.
  """

    class MinTlsVersionValueValuesEnum(_messages.Enum):
        """The minimum version of SSL protocol that can be used by the clients to
    establish a connection with the load balancer. This can be one of TLS_1_0,
    TLS_1_1, TLS_1_2.

    Values:
      TLS_1_0: TLS 1.0
      TLS_1_1: TLS 1.1
      TLS_1_2: TLS 1.2
    """
        TLS_1_0 = 0
        TLS_1_1 = 1
        TLS_1_2 = 2

    class ProfileValueValuesEnum(_messages.Enum):
        """Profile specifies the set of SSL features that can be used by the load
    balancer when negotiating SSL with clients. This can be one of COMPATIBLE,
    MODERN, RESTRICTED, or CUSTOM. If using CUSTOM, the set of SSL features to
    enable must be specified in the customFeatures field.

    Values:
      COMPATIBLE: Compatible profile. Allows the broadset set of clients, even
        those which support only out-of-date SSL features to negotiate with
        the load balancer.
      CUSTOM: Custom profile. Allow only the set of allowed SSL features
        specified in the customFeatures field.
      MODERN: Modern profile. Supports a wide set of SSL features, allowing
        modern clients to negotiate SSL with the load balancer.
      RESTRICTED: Restricted profile. Supports a reduced set of SSL features,
        intended to meet stricter compliance requirements.
    """
        COMPATIBLE = 0
        CUSTOM = 1
        MODERN = 2
        RESTRICTED = 3

    class WarningsValueListEntry(_messages.Message):
        """A WarningsValueListEntry object.

    Enums:
      CodeValueValuesEnum: [Output Only] A warning code, if applicable. For
        example, Compute Engine returns NO_RESULTS_ON_PAGE if there are no
        results in the response.

    Messages:
      DataValueListEntry: A DataValueListEntry object.

    Fields:
      code: [Output Only] A warning code, if applicable. For example, Compute
        Engine returns NO_RESULTS_ON_PAGE if there are no results in the
        response.
      data: [Output Only] Metadata about this warning in key: value format.
        For example: "data": [ { "key": "scope", "value": "zones/us-east1-d" }
      message: [Output Only] A human-readable description of the warning code.
    """

        class CodeValueValuesEnum(_messages.Enum):
            """[Output Only] A warning code, if applicable. For example, Compute
      Engine returns NO_RESULTS_ON_PAGE if there are no results in the
      response.

      Values:
        CLEANUP_FAILED: Warning about failed cleanup of transient changes made
          by a failed operation.
        DEPRECATED_RESOURCE_USED: A link to a deprecated resource was created.
        DEPRECATED_TYPE_USED: When deploying and at least one of the resources
          has a type marked as deprecated
        DISK_SIZE_LARGER_THAN_IMAGE_SIZE: The user created a boot disk that is
          larger than image size.
        EXPERIMENTAL_TYPE_USED: When deploying and at least one of the
          resources has a type marked as experimental
        EXTERNAL_API_WARNING: Warning that is present in an external api call
        FIELD_VALUE_OVERRIDEN: Warning that value of a field has been
          overridden. Deprecated unused field.
        INJECTED_KERNELS_DEPRECATED: The operation involved use of an injected
          kernel, which is deprecated.
        INVALID_HEALTH_CHECK_FOR_DYNAMIC_WIEGHTED_LB: A WEIGHTED_MAGLEV
          backend service is associated with a health check that is not of
          type HTTP/HTTPS/HTTP2.
        LARGE_DEPLOYMENT_WARNING: When deploying a deployment with a
          exceedingly large number of resources
        LIST_OVERHEAD_QUOTA_EXCEED: Resource can't be retrieved due to list
          overhead quota exceed which captures the amount of resources
          filtered out by user-defined list filter.
        MISSING_TYPE_DEPENDENCY: A resource depends on a missing type
        NEXT_HOP_ADDRESS_NOT_ASSIGNED: The route's nextHopIp address is not
          assigned to an instance on the network.
        NEXT_HOP_CANNOT_IP_FORWARD: The route's next hop instance cannot ip
          forward.
        NEXT_HOP_INSTANCE_HAS_NO_IPV6_INTERFACE: The route's nextHopInstance
          URL refers to an instance that does not have an ipv6 interface on
          the same network as the route.
        NEXT_HOP_INSTANCE_NOT_FOUND: The route's nextHopInstance URL refers to
          an instance that does not exist.
        NEXT_HOP_INSTANCE_NOT_ON_NETWORK: The route's nextHopInstance URL
          refers to an instance that is not on the same network as the route.
        NEXT_HOP_NOT_RUNNING: The route's next hop instance does not have a
          status of RUNNING.
        NOT_CRITICAL_ERROR: Error which is not critical. We decided to
          continue the process despite the mentioned error.
        NO_RESULTS_ON_PAGE: No results are present on a particular list page.
        PARTIAL_SUCCESS: Success is reported, but some results may be missing
          due to errors
        REQUIRED_TOS_AGREEMENT: The user attempted to use a resource that
          requires a TOS they have not accepted.
        RESOURCE_IN_USE_BY_OTHER_RESOURCE_WARNING: Warning that a resource is
          in use.
        RESOURCE_NOT_DELETED: One or more of the resources set to auto-delete
          could not be deleted because they were in use.
        SCHEMA_VALIDATION_IGNORED: When a resource schema validation is
          ignored.
        SINGLE_INSTANCE_PROPERTY_TEMPLATE: Instance template used in instance
          group manager is valid as such, but its application does not make a
          lot of sense, because it allows only single instance in instance
          group.
        UNDECLARED_PROPERTIES: When undeclared properties in the schema are
          present
        UNREACHABLE: A given scope cannot be reached.
      """
            CLEANUP_FAILED = 0
            DEPRECATED_RESOURCE_USED = 1
            DEPRECATED_TYPE_USED = 2
            DISK_SIZE_LARGER_THAN_IMAGE_SIZE = 3
            EXPERIMENTAL_TYPE_USED = 4
            EXTERNAL_API_WARNING = 5
            FIELD_VALUE_OVERRIDEN = 6
            INJECTED_KERNELS_DEPRECATED = 7
            INVALID_HEALTH_CHECK_FOR_DYNAMIC_WIEGHTED_LB = 8
            LARGE_DEPLOYMENT_WARNING = 9
            LIST_OVERHEAD_QUOTA_EXCEED = 10
            MISSING_TYPE_DEPENDENCY = 11
            NEXT_HOP_ADDRESS_NOT_ASSIGNED = 12
            NEXT_HOP_CANNOT_IP_FORWARD = 13
            NEXT_HOP_INSTANCE_HAS_NO_IPV6_INTERFACE = 14
            NEXT_HOP_INSTANCE_NOT_FOUND = 15
            NEXT_HOP_INSTANCE_NOT_ON_NETWORK = 16
            NEXT_HOP_NOT_RUNNING = 17
            NOT_CRITICAL_ERROR = 18
            NO_RESULTS_ON_PAGE = 19
            PARTIAL_SUCCESS = 20
            REQUIRED_TOS_AGREEMENT = 21
            RESOURCE_IN_USE_BY_OTHER_RESOURCE_WARNING = 22
            RESOURCE_NOT_DELETED = 23
            SCHEMA_VALIDATION_IGNORED = 24
            SINGLE_INSTANCE_PROPERTY_TEMPLATE = 25
            UNDECLARED_PROPERTIES = 26
            UNREACHABLE = 27

        class DataValueListEntry(_messages.Message):
            """A DataValueListEntry object.

      Fields:
        key: [Output Only] A key that provides more detail on the warning
          being returned. For example, for warnings where there are no results
          in a list request for a particular zone, this key might be scope and
          the key value might be the zone name. Other examples might be a key
          indicating a deprecated resource and a suggested replacement, or a
          warning about invalid network settings (for example, if an instance
          attempts to perform IP forwarding but is not enabled for IP
          forwarding).
        value: [Output Only] A warning data value corresponding to the key.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        code = _messages.EnumField('CodeValueValuesEnum', 1)
        data = _messages.MessageField('DataValueListEntry', 2, repeated=True)
        message = _messages.StringField(3)
    creationTimestamp = _messages.StringField(1)
    customFeatures = _messages.StringField(2, repeated=True)
    description = _messages.StringField(3)
    enabledFeatures = _messages.StringField(4, repeated=True)
    fingerprint = _messages.BytesField(5)
    id = _messages.IntegerField(6, variant=_messages.Variant.UINT64)
    kind = _messages.StringField(7, default='compute#sslPolicy')
    minTlsVersion = _messages.EnumField('MinTlsVersionValueValuesEnum', 8)
    name = _messages.StringField(9)
    profile = _messages.EnumField('ProfileValueValuesEnum', 10)
    region = _messages.StringField(11)
    selfLink = _messages.StringField(12)
    warnings = _messages.MessageField('WarningsValueListEntry', 13, repeated=True)