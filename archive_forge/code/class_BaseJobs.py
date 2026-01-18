from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
class BaseJobs(object):
    """Base API client for Cloud Scheduler jobs."""

    def __init__(self, messages, jobs_service, legacy_cron):
        self.messages = messages
        self.jobs_service = jobs_service
        self.legacy_cron = legacy_cron

    def List(self, parent_ref, limit=None, page_size=100):
        request = self.messages.CloudschedulerProjectsLocationsJobsListRequest(parent=parent_ref, legacyAppEngineCron=self.legacy_cron)
        return list_pager.YieldFromList(self.jobs_service, request, batch_size=page_size, limit=limit, field='jobs', batch_size_attribute='pageSize')

    def Delete(self, job_ref):
        request = self.messages.CloudschedulerProjectsLocationsJobsDeleteRequest(name=job_ref, legacyAppEngineCron=self.legacy_cron)
        return self.jobs_service.Delete(request)

    def Create(self, parent_ref, job):
        request = self.messages.CloudschedulerProjectsLocationsJobsCreateRequest(job=job, parent=parent_ref)
        return self.jobs_service.Create(request)

    def Run(self, job_ref, legacy_cron=False):
        request = self.messages.CloudschedulerProjectsLocationsJobsRunRequest(name=job_ref, runJobRequest=self.messages.RunJobRequest(legacyAppEngineCron=legacy_cron))
        return self.jobs_service.Run(request)