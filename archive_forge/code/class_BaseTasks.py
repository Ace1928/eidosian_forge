from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.core import exceptions
class BaseTasks(object):
    """API client for Cloud Tasks tasks."""

    def __init__(self, messages, tasks_service):
        self.messages = messages
        self.tasks_service = tasks_service

    def List(self, parent_ref, limit=None, page_size=100):
        request = self.messages.CloudtasksProjectsLocationsQueuesTasksListRequest(parent=parent_ref.RelativeName())
        return list_pager.YieldFromList(self.tasks_service, request, batch_size=page_size, limit=limit, field='tasks', batch_size_attribute='pageSize')

    def Get(self, task_ref, response_view=None):
        request = self.messages.CloudtasksProjectsLocationsQueuesTasksGetRequest(name=task_ref.RelativeName(), responseView=response_view)
        return self.tasks_service.Get(request)

    def Delete(self, task_ref):
        request = self.messages.CloudtasksProjectsLocationsQueuesTasksDeleteRequest(name=task_ref.RelativeName())
        return self.tasks_service.Delete(request)

    def Run(self, task_ref):
        request = self.messages.CloudtasksProjectsLocationsQueuesTasksRunRequest(name=task_ref.RelativeName())
        return self.tasks_service.Run(request)