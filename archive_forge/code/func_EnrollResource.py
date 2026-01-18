from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.auditmanager.v1alpha import auditmanager_v1alpha_messages as messages
def EnrollResource(self, request, global_params=None):
    """Enrolls the customer resource(folder/project) to the audit manager service by creating the audit managers P4SA in customers workload and granting required permissions to the P4SA. Please note that if enrollment request is made on the already enrolled workload then enrollment is executed overriding the existing set of destinations. As per https://google.aip.dev/127 recommendation, we are having multiple URI binding for Enroll API.

      Args:
        request: (AuditmanagerProjectsLocationsEnrollResourceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Enrollment) The response message.
      """
    config = self.GetMethodConfig('EnrollResource')
    return self._RunMethod(config, request, global_params=global_params)