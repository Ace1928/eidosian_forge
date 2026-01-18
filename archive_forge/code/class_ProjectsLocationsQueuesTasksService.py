from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudtasks.v2 import cloudtasks_v2_messages as messages
class ProjectsLocationsQueuesTasksService(base_api.BaseApiService):
    """Service class for the projects_locations_queues_tasks resource."""
    _NAME = 'projects_locations_queues_tasks'

    def __init__(self, client):
        super(CloudtasksV2.ProjectsLocationsQueuesTasksService, self).__init__(client)
        self._upload_configs = {}

    def Buffer(self, request, global_params=None):
        """Creates and buffers a new task without the need to explicitly define a Task message. The queue must have HTTP target. To create the task with a custom ID, use the following format and set TASK_ID to your desired ID: projects/PROJECT_ID/locations/LOCATION_ID/queues/QUEUE_ID/tasks/TASK_ID:buffer To create the task with an automatically generated ID, use the following format: projects/PROJECT_ID/locations/LOCATION_ID/queues/QUEUE_ID/tasks:buffer.

      Args:
        request: (CloudtasksProjectsLocationsQueuesTasksBufferRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BufferTaskResponse) The response message.
      """
        config = self.GetMethodConfig('Buffer')
        return self._RunMethod(config, request, global_params=global_params)
    Buffer.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/queues/{queuesId}/tasks/{taskId}:buffer', http_method='POST', method_id='cloudtasks.projects.locations.queues.tasks.buffer', ordered_params=['queue', 'taskId'], path_params=['queue', 'taskId'], query_params=[], relative_path='v2/{+queue}/tasks/{taskId}:buffer', request_field='bufferTaskRequest', request_type_name='CloudtasksProjectsLocationsQueuesTasksBufferRequest', response_type_name='BufferTaskResponse', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a task and adds it to a queue. Tasks cannot be updated after creation; there is no UpdateTask command. * The maximum task size is 100KB.

      Args:
        request: (CloudtasksProjectsLocationsQueuesTasksCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Task) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/queues/{queuesId}/tasks', http_method='POST', method_id='cloudtasks.projects.locations.queues.tasks.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/tasks', request_field='createTaskRequest', request_type_name='CloudtasksProjectsLocationsQueuesTasksCreateRequest', response_type_name='Task', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a task. A task can be deleted if it is scheduled or dispatched. A task cannot be deleted if it has executed successfully or permanently failed.

      Args:
        request: (CloudtasksProjectsLocationsQueuesTasksDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/queues/{queuesId}/tasks/{tasksId}', http_method='DELETE', method_id='cloudtasks.projects.locations.queues.tasks.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='CloudtasksProjectsLocationsQueuesTasksDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a task.

      Args:
        request: (CloudtasksProjectsLocationsQueuesTasksGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Task) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/queues/{queuesId}/tasks/{tasksId}', http_method='GET', method_id='cloudtasks.projects.locations.queues.tasks.get', ordered_params=['name'], path_params=['name'], query_params=['responseView'], relative_path='v2/{+name}', request_field='', request_type_name='CloudtasksProjectsLocationsQueuesTasksGetRequest', response_type_name='Task', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the tasks in a queue. By default, only the BASIC view is retrieved due to performance considerations; response_view controls the subset of information which is returned. The tasks may be returned in any order. The ordering may change at any time.

      Args:
        request: (CloudtasksProjectsLocationsQueuesTasksListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListTasksResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/queues/{queuesId}/tasks', http_method='GET', method_id='cloudtasks.projects.locations.queues.tasks.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken', 'responseView'], relative_path='v2/{+parent}/tasks', request_field='', request_type_name='CloudtasksProjectsLocationsQueuesTasksListRequest', response_type_name='ListTasksResponse', supports_download=False)

    def Run(self, request, global_params=None):
        """Forces a task to run now. When this method is called, Cloud Tasks will dispatch the task, even if the task is already running, the queue has reached its RateLimits or is PAUSED. This command is meant to be used for manual debugging. For example, RunTask can be used to retry a failed task after a fix has been made or to manually force a task to be dispatched now. The dispatched task is returned. That is, the task that is returned contains the status after the task is dispatched but before the task is received by its target. If Cloud Tasks receives a successful response from the task's target, then the task will be deleted; otherwise the task's schedule_time will be reset to the time that RunTask was called plus the retry delay specified in the queue's RetryConfig. RunTask returns NOT_FOUND when it is called on a task that has already succeeded or permanently failed.

      Args:
        request: (CloudtasksProjectsLocationsQueuesTasksRunRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Task) The response message.
      """
        config = self.GetMethodConfig('Run')
        return self._RunMethod(config, request, global_params=global_params)
    Run.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/queues/{queuesId}/tasks/{tasksId}:run', http_method='POST', method_id='cloudtasks.projects.locations.queues.tasks.run', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}:run', request_field='runTaskRequest', request_type_name='CloudtasksProjectsLocationsQueuesTasksRunRequest', response_type_name='Task', supports_download=False)