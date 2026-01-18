from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsConsentStoresEvaluateUserConsentsRequest(_messages.Message):
    """A
  HealthcareProjectsLocationsDatasetsConsentStoresEvaluateUserConsentsRequest
  object.

  Fields:
    consentStore: Required. Name of the consent store to retrieve User data
      mappings from.
    evaluateUserConsentsRequest: A EvaluateUserConsentsRequest resource to be
      passed as the request body.
  """
    consentStore = _messages.StringField(1, required=True)
    evaluateUserConsentsRequest = _messages.MessageField('EvaluateUserConsentsRequest', 2)