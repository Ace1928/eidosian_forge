from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecommenderV1alpha2SustainabilityProjection(_messages.Message):
    """Contains metadata about how much sustainability a recommendation can
  save or incur.

  Fields:
    duration: Duration for which this sustanability applies.
    kgCO2e: Carbon Footprint generated in kg of CO2 equivalent. Chose kg_c_o2e
      so that the name renders correctly in camelCase (kgCO2e).
  """
    duration = _messages.StringField(1)
    kgCO2e = _messages.FloatField(2)