from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CloudquotasProjectsLocationsServicesQuotaInfosGetRequest(_messages.Message):
    """A CloudquotasProjectsLocationsServicesQuotaInfosGetRequest object.

  Fields:
    name: Required. The resource name of the quota info. An example name: `pro
      jects/123/locations/global/services/compute.googleapis.com/quotaInfos/Cp
      usPerProjectPerRegion`
  """
    name = _messages.StringField(1, required=True)