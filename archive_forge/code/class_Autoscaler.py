from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Autoscaler(_messages.Message):
    """Represents an Autoscaler resource. Google Compute Engine has two
  Autoscaler resources: *
  [Zonal](/compute/docs/reference/rest/beta/autoscalers) *
  [Regional](/compute/docs/reference/rest/beta/regionAutoscalers) Use
  autoscalers to automatically add or delete instances from a managed instance
  group according to your defined autoscaling policy. For more information,
  read Autoscaling Groups of Instances. For zonal managed instance groups
  resource, use the autoscaler resource. For regional managed instance groups,
  use the regionAutoscalers resource.

  Enums:
    StatusValueValuesEnum: [Output Only] The status of the autoscaler
      configuration. Current set of possible values: - PENDING: Autoscaler
      backend hasn't read new/updated configuration. - DELETING: Configuration
      is being deleted. - ACTIVE: Configuration is acknowledged to be
      effective. Some warnings might be present in the statusDetails field. -
      ERROR: Configuration has errors. Actionable for users. Details are
      present in the statusDetails field. New values might be added in the
      future.

  Messages:
    ScalingScheduleStatusValue: [Output Only] Status information of existing
      scaling schedules.

  Fields:
    autoscalingPolicy: The configuration parameters for the autoscaling
      algorithm. You can define one or more signals for an autoscaler:
      cpuUtilization, customMetricUtilizations, and loadBalancingUtilization.
      If none of these are specified, the default will be to autoscale based
      on cpuUtilization to 0.6 or 60%.
    creationTimestamp: [Output Only] Creation timestamp in RFC3339 text
      format.
    description: An optional description of this resource. Provide this
      property when you create the resource.
    id: [Output Only] The unique identifier for the resource. This identifier
      is defined by the server.
    kind: [Output Only] Type of the resource. Always compute#autoscaler for
      autoscalers.
    name: Name of the resource. Provided by the client when the resource is
      created. The name must be 1-63 characters long, and comply with RFC1035.
      Specifically, the name must be 1-63 characters long and match the
      regular expression `[a-z]([-a-z0-9]*[a-z0-9])?` which means the first
      character must be a lowercase letter, and all following characters must
      be a dash, lowercase letter, or digit, except the last character, which
      cannot be a dash.
    recommendedSize: [Output Only] Target recommended MIG size (number of
      instances) computed by autoscaler. Autoscaler calculates the recommended
      MIG size even when the autoscaling policy mode is different from ON.
      This field is empty when autoscaler is not connected to an existing
      managed instance group or autoscaler did not generate its prediction.
    region: [Output Only] URL of the region where the instance group resides
      (for autoscalers living in regional scope).
    scalingScheduleStatus: [Output Only] Status information of existing
      scaling schedules.
    selfLink: [Output Only] Server-defined URL for the resource.
    status: [Output Only] The status of the autoscaler configuration. Current
      set of possible values: - PENDING: Autoscaler backend hasn't read
      new/updated configuration. - DELETING: Configuration is being deleted. -
      ACTIVE: Configuration is acknowledged to be effective. Some warnings
      might be present in the statusDetails field. - ERROR: Configuration has
      errors. Actionable for users. Details are present in the statusDetails
      field. New values might be added in the future.
    statusDetails: [Output Only] Human-readable details about the current
      state of the autoscaler. Read the documentation for Commonly returned
      status messages for examples of status messages you might encounter.
    target: URL of the managed instance group that this autoscaler will scale.
      This field is required when creating an autoscaler.
    zone: [Output Only] URL of the zone where the instance group resides (for
      autoscalers living in zonal scope).
  """

    class StatusValueValuesEnum(_messages.Enum):
        """[Output Only] The status of the autoscaler configuration. Current set
    of possible values: - PENDING: Autoscaler backend hasn't read new/updated
    configuration. - DELETING: Configuration is being deleted. - ACTIVE:
    Configuration is acknowledged to be effective. Some warnings might be
    present in the statusDetails field. - ERROR: Configuration has errors.
    Actionable for users. Details are present in the statusDetails field. New
    values might be added in the future.

    Values:
      ACTIVE: Configuration is acknowledged to be effective
      DELETING: Configuration is being deleted
      ERROR: Configuration has errors. Actionable for users.
      PENDING: Autoscaler backend hasn't read new/updated configuration
    """
        ACTIVE = 0
        DELETING = 1
        ERROR = 2
        PENDING = 3

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ScalingScheduleStatusValue(_messages.Message):
        """[Output Only] Status information of existing scaling schedules.

    Messages:
      AdditionalProperty: An additional property for a
        ScalingScheduleStatusValue object.

    Fields:
      additionalProperties: Additional properties of type
        ScalingScheduleStatusValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ScalingScheduleStatusValue object.

      Fields:
        key: Name of the additional property.
        value: A ScalingScheduleStatus attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('ScalingScheduleStatus', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    autoscalingPolicy = _messages.MessageField('AutoscalingPolicy', 1)
    creationTimestamp = _messages.StringField(2)
    description = _messages.StringField(3)
    id = _messages.IntegerField(4, variant=_messages.Variant.UINT64)
    kind = _messages.StringField(5, default='compute#autoscaler')
    name = _messages.StringField(6)
    recommendedSize = _messages.IntegerField(7, variant=_messages.Variant.INT32)
    region = _messages.StringField(8)
    scalingScheduleStatus = _messages.MessageField('ScalingScheduleStatusValue', 9)
    selfLink = _messages.StringField(10)
    status = _messages.EnumField('StatusValueValuesEnum', 11)
    statusDetails = _messages.MessageField('AutoscalerStatusDetails', 12, repeated=True)
    target = _messages.StringField(13)
    zone = _messages.StringField(14)