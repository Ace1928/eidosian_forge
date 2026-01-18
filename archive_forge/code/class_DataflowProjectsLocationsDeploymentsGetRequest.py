from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataflowProjectsLocationsDeploymentsGetRequest(_messages.Message):
    """A DataflowProjectsLocationsDeploymentsGetRequest object.

  Enums:
    ViewValueValuesEnum:

  Fields:
    name: Required. The name of the `Deployment`. Format:
      projects/{project}/locations/{location}/deployments/{deployment_id}
    view: A ViewValueValuesEnum attribute.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """ViewValueValuesEnum enum type.

    Values:
      DEPLOYMENT_VIEW_UNSPECIFIED: The deployment view to return isn't
        specified, or is unknown. Responses will contain at least the
        `DEPLOYMENT_VIEW_ALL` information, and may contain additional
        information.
      DEPLOYMENT_VIEW_ALL: Request all information available for this
        deployment.
      DEPLOYMENT_VIEW_MUTABLE: Request all mutable information available for
        this deployment.
    """
        DEPLOYMENT_VIEW_UNSPECIFIED = 0
        DEPLOYMENT_VIEW_ALL = 1
        DEPLOYMENT_VIEW_MUTABLE = 2
    name = _messages.StringField(1, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 2)