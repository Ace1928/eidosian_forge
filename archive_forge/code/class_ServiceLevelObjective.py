from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceLevelObjective(_messages.Message):
    """A Service-Level Objective (SLO) describes a level of desired good
  service. It consists of a service-level indicator (SLI), a performance goal,
  and a period over which the objective is to be evaluated against that goal.
  The SLO can use SLIs defined in a number of different manners. Typical SLOs
  might include "99% of requests in each rolling week have latency below 200
  milliseconds" or "99.5% of requests in each calendar month return
  successfully."

  Enums:
    CalendarPeriodValueValuesEnum: A calendar period, semantically "since the
      start of the current ". At this time, only DAY, WEEK, FORTNIGHT, and
      MONTH are supported.

  Messages:
    UserLabelsValue: Labels which have been used to annotate the service-level
      objective. Label keys must start with a letter. Label keys and values
      may contain lowercase letters, numbers, underscores, and dashes. Label
      keys and values have a maximum length of 63 characters, and must be less
      than 128 bytes in size. Up to 64 label entries may be stored. For labels
      which do not have a semantic value, the empty string may be supplied for
      the label value.

  Fields:
    calendarPeriod: A calendar period, semantically "since the start of the
      current ". At this time, only DAY, WEEK, FORTNIGHT, and MONTH are
      supported.
    displayName: Name used for UI elements listing this SLO.
    goal: The fraction of service that must be good in order for this
      objective to be met. 0 < goal <= 0.999.
    name: Resource name for this ServiceLevelObjective. The format is: project
      s/[PROJECT_ID_OR_NUMBER]/services/[SERVICE_ID]/serviceLevelObjectives/[S
      LO_NAME]
    rollingPeriod: A rolling time period, semantically "in the past ". Must be
      an integer multiple of 1 day no larger than 30 days.
    serviceLevelIndicator: The definition of good service, used to measure and
      calculate the quality of the Service's performance with respect to a
      single aspect of service quality.
    userLabels: Labels which have been used to annotate the service-level
      objective. Label keys must start with a letter. Label keys and values
      may contain lowercase letters, numbers, underscores, and dashes. Label
      keys and values have a maximum length of 63 characters, and must be less
      than 128 bytes in size. Up to 64 label entries may be stored. For labels
      which do not have a semantic value, the empty string may be supplied for
      the label value.
  """

    class CalendarPeriodValueValuesEnum(_messages.Enum):
        """A calendar period, semantically "since the start of the current ". At
    this time, only DAY, WEEK, FORTNIGHT, and MONTH are supported.

    Values:
      CALENDAR_PERIOD_UNSPECIFIED: Undefined period, raises an error.
      DAY: A day.
      WEEK: A week. Weeks begin on Monday, following ISO 8601
        (https://en.wikipedia.org/wiki/ISO_week_date).
      FORTNIGHT: A fortnight. The first calendar fortnight of the year begins
        at the start of week 1 according to ISO 8601
        (https://en.wikipedia.org/wiki/ISO_week_date).
      MONTH: A month.
      QUARTER: A quarter. Quarters start on dates 1-Jan, 1-Apr, 1-Jul, and
        1-Oct of each year.
      HALF: A half-year. Half-years start on dates 1-Jan and 1-Jul.
      YEAR: A year.
    """
        CALENDAR_PERIOD_UNSPECIFIED = 0
        DAY = 1
        WEEK = 2
        FORTNIGHT = 3
        MONTH = 4
        QUARTER = 5
        HALF = 6
        YEAR = 7

    @encoding.MapUnrecognizedFields('additionalProperties')
    class UserLabelsValue(_messages.Message):
        """Labels which have been used to annotate the service-level objective.
    Label keys must start with a letter. Label keys and values may contain
    lowercase letters, numbers, underscores, and dashes. Label keys and values
    have a maximum length of 63 characters, and must be less than 128 bytes in
    size. Up to 64 label entries may be stored. For labels which do not have a
    semantic value, the empty string may be supplied for the label value.

    Messages:
      AdditionalProperty: An additional property for a UserLabelsValue object.

    Fields:
      additionalProperties: Additional properties of type UserLabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a UserLabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    calendarPeriod = _messages.EnumField('CalendarPeriodValueValuesEnum', 1)
    displayName = _messages.StringField(2)
    goal = _messages.FloatField(3)
    name = _messages.StringField(4)
    rollingPeriod = _messages.StringField(5)
    serviceLevelIndicator = _messages.MessageField('ServiceLevelIndicator', 6)
    userLabels = _messages.MessageField('UserLabelsValue', 7)