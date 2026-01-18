from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ErrorGroup(_messages.Message):
    """Description of a group of similar error events.

  Enums:
    ResolutionStatusValueValuesEnum: Error group's resolution status. An
      unspecified resolution status will be interpreted as OPEN

  Fields:
    groupId: An opaque identifier of the group. This field is assigned by the
      Error Reporting system and always populated. In the group resource name,
      the `group_id` is a unique identifier for a particular error group. The
      identifier is derived from key parts of the error-log content and is
      treated as Service Data. For information about how Service Data is
      handled, see [Google Cloud Privacy
      Notice](https://cloud.google.com/terms/cloud-privacy-notice).
    name: The group resource name. Written as
      `projects/{projectID}/groups/{group_id}`. Example: `projects/my-
      project-123/groups/my-group` In the group resource name, the `group_id`
      is a unique identifier for a particular error group. The identifier is
      derived from key parts of the error-log content and is treated as
      Service Data. For information about how Service Data is handled, see
      [Google Cloud Privacy Notice](https://cloud.google.com/terms/cloud-
      privacy-notice).
    resolutionStatus: Error group's resolution status. An unspecified
      resolution status will be interpreted as OPEN
    trackingIssues: Associated tracking issues.
  """

    class ResolutionStatusValueValuesEnum(_messages.Enum):
        """Error group's resolution status. An unspecified resolution status will
    be interpreted as OPEN

    Values:
      RESOLUTION_STATUS_UNSPECIFIED: Status is unknown. When left unspecified
        in requests, it is treated like OPEN.
      OPEN: The error group is not being addressed. This is the default for
        new groups. It is also used for errors re-occurring after marked
        RESOLVED.
      ACKNOWLEDGED: Error Group manually acknowledged, it can have an issue
        link attached.
      RESOLVED: Error Group manually resolved, more events for this group are
        not expected to occur.
      MUTED: The error group is muted and excluded by default on group stats
        requests.
    """
        RESOLUTION_STATUS_UNSPECIFIED = 0
        OPEN = 1
        ACKNOWLEDGED = 2
        RESOLVED = 3
        MUTED = 4
    groupId = _messages.StringField(1)
    name = _messages.StringField(2)
    resolutionStatus = _messages.EnumField('ResolutionStatusValueValuesEnum', 3)
    trackingIssues = _messages.MessageField('TrackingIssue', 4, repeated=True)