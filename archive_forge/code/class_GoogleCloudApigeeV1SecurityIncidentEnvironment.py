from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1SecurityIncidentEnvironment(_messages.Message):
    """Represents an SecurityIncidentEnvironment resource.

  Fields:
    environment: Output only. Name of the environment
    lowRiskIncidentsCount: Output only. Total incidents with risk level low.
    moderateRiskIncidentsCount: Output only. Total incidents with risk level
      moderate.
    severeRiskIncidentsCount: Output only. Total incidents with risk level
      severe.
    totalIncidents: Output only. Total incidents count for a given environment
  """
    environment = _messages.StringField(1)
    lowRiskIncidentsCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    moderateRiskIncidentsCount = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    severeRiskIncidentsCount = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    totalIncidents = _messages.IntegerField(5, variant=_messages.Variant.INT32)