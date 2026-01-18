from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudtasks.v2 import cloudtasks_v2_messages as messages
class ProjectsLocationsQueuesService(base_api.BaseApiService):
    """Service class for the projects_locations_queues resource."""
    _NAME = 'projects_locations_queues'

    def __init__(self, client):
        super(CloudtasksV2.ProjectsLocationsQueuesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a queue. Queues created with this method allow tasks to live for a maximum of 31 days. After a task is 31 days old, the task will be deleted regardless of whether it was dispatched or not. WARNING: Using this method may have unintended side effects if you are using an App Engine `queue.yaml` or `queue.xml` file to manage your queues. Read [Overview of Queue Management and queue.yaml](https://cloud.google.com/tasks/docs/queue-yaml) before using this method.

      Args:
        request: (CloudtasksProjectsLocationsQueuesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Queue) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/queues', http_method='POST', method_id='cloudtasks.projects.locations.queues.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/queues', request_field='queue', request_type_name='CloudtasksProjectsLocationsQueuesCreateRequest', response_type_name='Queue', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a queue. This command will delete the queue even if it has tasks in it. Note: If you delete a queue, you may be prevented from creating a new queue with the same name as the deleted queue for a tombstone window of up to 3 days. During this window, the CreateQueue operation may appear to recreate the queue, but this can be misleading. If you attempt to create a queue with the same name as one that is in the tombstone window, run GetQueue to confirm that the queue creation was successful. If GetQueue returns 200 response code, your queue was successfully created with the name of the previously deleted queue. Otherwise, your queue did not successfully recreate. WARNING: Using this method may have unintended side effects if you are using an App Engine `queue.yaml` or `queue.xml` file to manage your queues. Read [Overview of Queue Management and queue.yaml](https://cloud.google.com/tasks/docs/queue-yaml) before using this method.

      Args:
        request: (CloudtasksProjectsLocationsQueuesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/queues/{queuesId}', http_method='DELETE', method_id='cloudtasks.projects.locations.queues.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='CloudtasksProjectsLocationsQueuesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a queue.

      Args:
        request: (CloudtasksProjectsLocationsQueuesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Queue) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/queues/{queuesId}', http_method='GET', method_id='cloudtasks.projects.locations.queues.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='CloudtasksProjectsLocationsQueuesGetRequest', response_type_name='Queue', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a Queue. Returns an empty policy if the resource exists and does not have a policy set. Authorization requires the following [Google IAM](https://cloud.google.com/iam) permission on the specified resource parent: * `cloudtasks.queues.getIamPolicy`.

      Args:
        request: (CloudtasksProjectsLocationsQueuesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/queues/{queuesId}:getIamPolicy', http_method='POST', method_id='cloudtasks.projects.locations.queues.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v2/{+resource}:getIamPolicy', request_field='getIamPolicyRequest', request_type_name='CloudtasksProjectsLocationsQueuesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists queues. Queues are returned in lexicographical order.

      Args:
        request: (CloudtasksProjectsLocationsQueuesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListQueuesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/queues', http_method='GET', method_id='cloudtasks.projects.locations.queues.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v2/{+parent}/queues', request_field='', request_type_name='CloudtasksProjectsLocationsQueuesListRequest', response_type_name='ListQueuesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a queue. This method creates the queue if it does not exist and updates the queue if it does exist. Queues created with this method allow tasks to live for a maximum of 31 days. After a task is 31 days old, the task will be deleted regardless of whether it was dispatched or not. WARNING: Using this method may have unintended side effects if you are using an App Engine `queue.yaml` or `queue.xml` file to manage your queues. Read [Overview of Queue Management and queue.yaml](https://cloud.google.com/tasks/docs/queue-yaml) before using this method.

      Args:
        request: (CloudtasksProjectsLocationsQueuesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Queue) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/queues/{queuesId}', http_method='PATCH', method_id='cloudtasks.projects.locations.queues.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v2/{+name}', request_field='queue', request_type_name='CloudtasksProjectsLocationsQueuesPatchRequest', response_type_name='Queue', supports_download=False)

    def Pause(self, request, global_params=None):
        """Pauses the queue. If a queue is paused then the system will stop dispatching tasks until the queue is resumed via ResumeQueue. Tasks can still be added when the queue is paused. A queue is paused if its state is PAUSED.

      Args:
        request: (CloudtasksProjectsLocationsQueuesPauseRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Queue) The response message.
      """
        config = self.GetMethodConfig('Pause')
        return self._RunMethod(config, request, global_params=global_params)
    Pause.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/queues/{queuesId}:pause', http_method='POST', method_id='cloudtasks.projects.locations.queues.pause', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}:pause', request_field='pauseQueueRequest', request_type_name='CloudtasksProjectsLocationsQueuesPauseRequest', response_type_name='Queue', supports_download=False)

    def Purge(self, request, global_params=None):
        """Purges a queue by deleting all of its tasks. All tasks created before this method is called are permanently deleted. Purge operations can take up to one minute to take effect. Tasks might be dispatched before the purge takes effect. A purge is irreversible.

      Args:
        request: (CloudtasksProjectsLocationsQueuesPurgeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Queue) The response message.
      """
        config = self.GetMethodConfig('Purge')
        return self._RunMethod(config, request, global_params=global_params)
    Purge.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/queues/{queuesId}:purge', http_method='POST', method_id='cloudtasks.projects.locations.queues.purge', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}:purge', request_field='purgeQueueRequest', request_type_name='CloudtasksProjectsLocationsQueuesPurgeRequest', response_type_name='Queue', supports_download=False)

    def Resume(self, request, global_params=None):
        """Resume a queue. This method resumes a queue after it has been PAUSED or DISABLED. The state of a queue is stored in the queue's state; after calling this method it will be set to RUNNING. WARNING: Resuming many high-QPS queues at the same time can lead to target overloading. If you are resuming high-QPS queues, follow the 500/50/5 pattern described in [Managing Cloud Tasks Scaling Risks](https://cloud.google.com/tasks/docs/manage-cloud-task-scaling).

      Args:
        request: (CloudtasksProjectsLocationsQueuesResumeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Queue) The response message.
      """
        config = self.GetMethodConfig('Resume')
        return self._RunMethod(config, request, global_params=global_params)
    Resume.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/queues/{queuesId}:resume', http_method='POST', method_id='cloudtasks.projects.locations.queues.resume', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}:resume', request_field='resumeQueueRequest', request_type_name='CloudtasksProjectsLocationsQueuesResumeRequest', response_type_name='Queue', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy for a Queue. Replaces any existing policy. Note: The Cloud Console does not check queue-level IAM permissions yet. Project-level permissions are required to use the Cloud Console. Authorization requires the following [Google IAM](https://cloud.google.com/iam) permission on the specified resource parent: * `cloudtasks.queues.setIamPolicy`.

      Args:
        request: (CloudtasksProjectsLocationsQueuesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/queues/{queuesId}:setIamPolicy', http_method='POST', method_id='cloudtasks.projects.locations.queues.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v2/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='CloudtasksProjectsLocationsQueuesSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on a Queue. If the resource does not exist, this will return an empty set of permissions, not a NOT_FOUND error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (CloudtasksProjectsLocationsQueuesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/queues/{queuesId}:testIamPermissions', http_method='POST', method_id='cloudtasks.projects.locations.queues.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v2/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='CloudtasksProjectsLocationsQueuesTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)