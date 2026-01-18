from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringServicesServiceLevelObjectivesDeleteRequest(_messages.Message):
    """A MonitoringServicesServiceLevelObjectivesDeleteRequest object.

  Fields:
    name: Required. Resource name of the ServiceLevelObjective to delete. The
      format is: projects/[PROJECT_ID_OR_NUMBER]/services/[SERVICE_ID]/service
      LevelObjectives/[SLO_NAME]
  """
    name = _messages.StringField(1, required=True)