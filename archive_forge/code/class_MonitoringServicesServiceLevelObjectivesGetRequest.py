from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringServicesServiceLevelObjectivesGetRequest(_messages.Message):
    """A MonitoringServicesServiceLevelObjectivesGetRequest object.

  Enums:
    ViewValueValuesEnum: View of the ServiceLevelObjective to return. If
      DEFAULT, return the ServiceLevelObjective as originally defined. If
      EXPLICIT and the ServiceLevelObjective is defined in terms of a
      BasicSli, replace the BasicSli with a RequestBasedSli spelling out how
      the SLI is computed.

  Fields:
    name: Required. Resource name of the ServiceLevelObjective to get. The
      format is: projects/[PROJECT_ID_OR_NUMBER]/services/[SERVICE_ID]/service
      LevelObjectives/[SLO_NAME]
    view: View of the ServiceLevelObjective to return. If DEFAULT, return the
      ServiceLevelObjective as originally defined. If EXPLICIT and the
      ServiceLevelObjective is defined in terms of a BasicSli, replace the
      BasicSli with a RequestBasedSli spelling out how the SLI is computed.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """View of the ServiceLevelObjective to return. If DEFAULT, return the
    ServiceLevelObjective as originally defined. If EXPLICIT and the
    ServiceLevelObjective is defined in terms of a BasicSli, replace the
    BasicSli with a RequestBasedSli spelling out how the SLI is computed.

    Values:
      VIEW_UNSPECIFIED: Same as FULL.
      FULL: Return the embedded ServiceLevelIndicator in the form in which it
        was defined. If it was defined using a BasicSli, return that BasicSli.
      EXPLICIT: For ServiceLevelIndicators using BasicSli articulation,
        instead return the ServiceLevelIndicator with its mode of computation
        fully spelled out as a RequestBasedSli. For ServiceLevelIndicators
        using RequestBasedSli or WindowsBasedSli, return the
        ServiceLevelIndicator as it was provided.
    """
        VIEW_UNSPECIFIED = 0
        FULL = 1
        EXPLICIT = 2
    name = _messages.StringField(1, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 2)