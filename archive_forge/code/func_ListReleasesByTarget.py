from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.clouddeploy import client_util
from googlecloudsdk.core import log
def ListReleasesByTarget(self, target_ref_project_number, project_id, pipeline_id):
    """Lists the releases in a target.

    Args:
      target_ref_project_number: target reference with project number in the
        name.
      project_id: str, project ID.
      pipeline_id: str, delivery pipeline ID.

    Returns:
      a list of release messages.
    """
    target_dict = target_ref_project_number.AsDict()
    request = self.messages.ClouddeployProjectsLocationsDeliveryPipelinesReleasesListRequest(parent=RELEASE_PARENT_TEMPLATE.format(project_id, target_dict['locationsId'], pipeline_id), filter=TARGET_FILTER_TEMPLATE.format(target_ref_project_number.RelativeName()))
    return self._service.List(request).releases