from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.scheduler import jobs
from googlecloudsdk.api_lib.scheduler import locations
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
class BetaApiAdapter(BaseApiAdapter):

    def __init__(self, legacy_cron=False):
        super(BetaApiAdapter, self).__init__(BETA_API_VERSION)
        self.jobs = jobs.BaseJobs(self.client.MESSAGES_MODULE, self.client.projects_locations_jobs, legacy_cron=legacy_cron)