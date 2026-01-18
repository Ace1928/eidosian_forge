from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringProjectsCollectdTimeSeriesCreateRequest(_messages.Message):
    """A MonitoringProjectsCollectdTimeSeriesCreateRequest object.

  Fields:
    createCollectdTimeSeriesRequest: A CreateCollectdTimeSeriesRequest
      resource to be passed as the request body.
    name: The project
      (https://cloud.google.com/monitoring/api/v3#project_name) in which to
      create the time series. The format is: projects/[PROJECT_ID_OR_NUMBER]
  """
    createCollectdTimeSeriesRequest = _messages.MessageField('CreateCollectdTimeSeriesRequest', 1)
    name = _messages.StringField(2, required=True)