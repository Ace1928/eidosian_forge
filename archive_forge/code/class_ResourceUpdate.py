from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourceUpdate(_messages.Message):
    """A ResourceUpdate object.

  Enums:
    IntentValueValuesEnum: Output only. The intent of the resource: `PREVIEW`,
      `UPDATE`, or `CANCEL`.
    StateValueValuesEnum: Output only. The state of the resource.

  Messages:
    ErrorValue: Output only. If errors are generated during update of the
      resource, this field will be populated.
    WarningsValueListEntry: A WarningsValueListEntry object.

  Fields:
    accessControl: The Access Control Policy to set on this resource after
      updating the resource itself.
    error: Output only. If errors are generated during update of the resource,
      this field will be populated.
    finalProperties: Output only. The expanded properties of the resource with
      reference values expanded. Returned as serialized YAML.
    intent: Output only. The intent of the resource: `PREVIEW`, `UPDATE`, or
      `CANCEL`.
    manifest: Output only. URL of the manifest representing the update
      configuration of this resource.
    properties: Output only. The set of updated properties for this resource,
      before references are expanded. Returned as serialized YAML.
    state: Output only. The state of the resource.
    warnings: Output only. If warning messages are generated during processing
      of this resource, this field will be populated.
  """

    class IntentValueValuesEnum(_messages.Enum):
        """Output only. The intent of the resource: `PREVIEW`, `UPDATE`, or
    `CANCEL`.

    Values:
      CREATE_OR_ACQUIRE: The resource is scheduled to be created, or if it
        already exists, acquired.
      DELETE: The resource is scheduled to be deleted.
      ACQUIRE: The resource is scheduled to be acquired.
      UPDATE: The resource is scheduled to be updated via the UPDATE method.
      ABANDON: The resource is scheduled to be abandoned.
      CREATE: The resource is scheduled to be created.
    """
        CREATE_OR_ACQUIRE = 0
        DELETE = 1
        ACQUIRE = 2
        UPDATE = 3
        ABANDON = 4
        CREATE = 5

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The state of the resource.

    Values:
      PENDING: There are changes pending for this resource.
      IN_PROGRESS: The service is executing changes on the resource.
      IN_PREVIEW: The service is previewing changes on the resource.
      FAILED: The service has failed to change the resource.
      ABORTED: The service has aborted trying to change the resource.
    """
        PENDING = 0
        IN_PROGRESS = 1
        IN_PREVIEW = 2
        FAILED = 3
        ABORTED = 4

    class ErrorValue(_messages.Message):
        """Output only. If errors are generated during update of the resource,
    this field will be populated.

    Messages:
      ErrorsValueListEntry: A ErrorsValueListEntry object.

    Fields:
      errors: [Output Only] The array of errors encountered while processing
        this operation.
    """

        class ErrorsValueListEntry(_messages.Message):
            """A ErrorsValueListEntry object.

      Fields:
        code: [Output Only] The error type identifier for this error.
        location: [Output Only] Indicates the field in the request that caused
          the error. This property is optional.
        message: [Output Only] An optional, human-readable error message.
      """
            code = _messages.StringField(1)
            location = _messages.StringField(2)
            message = _messages.StringField(3)
        errors = _messages.MessageField('ErrorsValueListEntry', 1, repeated=True)

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
        DEPRECATED_RESOURCE_USED: A link to a deprecated resource was created.
        NO_RESULTS_ON_PAGE: No results are present on a particular list page.
        UNREACHABLE: A given scope cannot be reached.
        NEXT_HOP_ADDRESS_NOT_ASSIGNED: The route's nextHopIp address is not
          assigned to an instance on the network.
        NEXT_HOP_INSTANCE_NOT_FOUND: The route's nextHopInstance URL refers to
          an instance that does not exist.
        NEXT_HOP_INSTANCE_NOT_ON_NETWORK: The route's nextHopInstance URL
          refers to an instance that is not on the same network as the route.
        NEXT_HOP_CANNOT_IP_FORWARD: The route's next hop instance cannot ip
          forward.
        NEXT_HOP_NOT_RUNNING: The route's next hop instance does not have a
          status of RUNNING.
        INJECTED_KERNELS_DEPRECATED: The operation involved use of an injected
          kernel, which is deprecated.
        REQUIRED_TOS_AGREEMENT: The user attempted to use a resource that
          requires a TOS they have not accepted.
        DISK_SIZE_LARGER_THAN_IMAGE_SIZE: The user created a boot disk that is
          larger than image size.
        RESOURCE_NOT_DELETED: One or more of the resources set to auto-delete
          could not be deleted because they were in use.
        SINGLE_INSTANCE_PROPERTY_TEMPLATE: Instance template used in instance
          group manager is valid as such, but its application does not make a
          lot of sense, because it allows only single instance in instance
          group.
        NOT_CRITICAL_ERROR: Error which is not critical. We decided to
          continue the process despite the mentioned error.
        CLEANUP_FAILED: Warning about failed cleanup of transient changes made
          by a failed operation.
        FIELD_VALUE_OVERRIDEN: Warning that value of a field has been
          overridden. Deprecated unused field.
        RESOURCE_IN_USE_BY_OTHER_RESOURCE_WARNING: Warning that a resource is
          in use.
        MISSING_TYPE_DEPENDENCY: A resource depends on a missing type
        EXTERNAL_API_WARNING: Warning that is present in an external api call
        SCHEMA_VALIDATION_IGNORED: When a resource schema validation is
          ignored.
        UNDECLARED_PROPERTIES: When undeclared properties in the schema are
          present
        EXPERIMENTAL_TYPE_USED: When deploying and at least one of the
          resources has a type marked as experimental
        DEPRECATED_TYPE_USED: When deploying and at least one of the resources
          has a type marked as deprecated
        PARTIAL_SUCCESS: Success is reported, but some results may be missing
          due to errors
        LARGE_DEPLOYMENT_WARNING: When deploying a deployment with a
          exceedingly large number of resources
        NEXT_HOP_INSTANCE_HAS_NO_IPV6_INTERFACE: The route's nextHopInstance
          URL refers to an instance that does not have an ipv6 interface on
          the same network as the route.
        INVALID_HEALTH_CHECK_FOR_DYNAMIC_WIEGHTED_LB: A WEIGHTED_MAGLEV
          backend service is associated with a health check that is not of
          type HTTP/HTTPS/HTTP2.
        LIST_OVERHEAD_QUOTA_EXCEED: Resource can't be retrieved due to list
          overhead quota exceed which captures the amount of resources
          filtered out by user-defined list filter.
      """
            DEPRECATED_RESOURCE_USED = 0
            NO_RESULTS_ON_PAGE = 1
            UNREACHABLE = 2
            NEXT_HOP_ADDRESS_NOT_ASSIGNED = 3
            NEXT_HOP_INSTANCE_NOT_FOUND = 4
            NEXT_HOP_INSTANCE_NOT_ON_NETWORK = 5
            NEXT_HOP_CANNOT_IP_FORWARD = 6
            NEXT_HOP_NOT_RUNNING = 7
            INJECTED_KERNELS_DEPRECATED = 8
            REQUIRED_TOS_AGREEMENT = 9
            DISK_SIZE_LARGER_THAN_IMAGE_SIZE = 10
            RESOURCE_NOT_DELETED = 11
            SINGLE_INSTANCE_PROPERTY_TEMPLATE = 12
            NOT_CRITICAL_ERROR = 13
            CLEANUP_FAILED = 14
            FIELD_VALUE_OVERRIDEN = 15
            RESOURCE_IN_USE_BY_OTHER_RESOURCE_WARNING = 16
            MISSING_TYPE_DEPENDENCY = 17
            EXTERNAL_API_WARNING = 18
            SCHEMA_VALIDATION_IGNORED = 19
            UNDECLARED_PROPERTIES = 20
            EXPERIMENTAL_TYPE_USED = 21
            DEPRECATED_TYPE_USED = 22
            PARTIAL_SUCCESS = 23
            LARGE_DEPLOYMENT_WARNING = 24
            NEXT_HOP_INSTANCE_HAS_NO_IPV6_INTERFACE = 25
            INVALID_HEALTH_CHECK_FOR_DYNAMIC_WIEGHTED_LB = 26
            LIST_OVERHEAD_QUOTA_EXCEED = 27

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
    accessControl = _messages.MessageField('ResourceAccessControl', 1)
    error = _messages.MessageField('ErrorValue', 2)
    finalProperties = _messages.StringField(3)
    intent = _messages.EnumField('IntentValueValuesEnum', 4)
    manifest = _messages.StringField(5)
    properties = _messages.StringField(6)
    state = _messages.EnumField('StateValueValuesEnum', 7)
    warnings = _messages.MessageField('WarningsValueListEntry', 8, repeated=True)