from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsDataScansGenerateDataQualityRulesRequest(_messages.Message):
    """A DataplexProjectsLocationsDataScansGenerateDataQualityRulesRequest
  object.

  Fields:
    googleCloudDataplexV1GenerateDataQualityRulesRequest: A
      GoogleCloudDataplexV1GenerateDataQualityRulesRequest resource to be
      passed as the request body.
    name: Required. The name should be either * the name of a datascan with at
      least one successful completed data profiling job, or * the name of a
      successful completed data profiling datascan job.
  """
    googleCloudDataplexV1GenerateDataQualityRulesRequest = _messages.MessageField('GoogleCloudDataplexV1GenerateDataQualityRulesRequest', 1)
    name = _messages.StringField(2, required=True)