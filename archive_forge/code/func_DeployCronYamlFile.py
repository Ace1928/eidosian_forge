from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.app import util
from googlecloudsdk.api_lib.tasks import task_queues_convertors as convertors
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.tasks import app
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.command_lib.tasks import flags
from googlecloudsdk.command_lib.tasks import parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
import six
from six.moves import urllib
def DeployCronYamlFile(scheduler_api, config, existing_jobs):
    """Perform a deployment based on the parsed 'cron.yaml' file.

  For every job defined in the cron.yaml file, we will create a new cron job
  for any job that did not already exist in our backend. We will also delete
  those jobs which are not present in the YAML file but exist in our backend.
  Note: We do not update any jobs. The only operations are Create and Delete.
  So if we modify any attribute of an existing job in the YAML file, the old
  job gets deleted and a new job is created based on the new attributes.

  Args:
    scheduler_api: api_lib.scheduler.<Alpha|Beta|GA>ApiAdapter, Cloud Scheduler
      API needed for doing jobs based operations.
    config: A yaml_parsing.ConfigYamlInfo object for the parsed YAML file we
      are going to process.
   existing_jobs: A list of jobs that already exist in the backend. Each job
      maps to an apis.cloudscheduler.<ver>.cloudscheduler<ver>_messages.Job
      instance.
  Returns:
    A list of responses received from the Cloud Scheduler APIs representing job
    states for every call made to create a job.
  """
    cron_yaml = config.parsed
    jobs_client = scheduler_api.jobs
    app_location = app.ResolveAppLocation(parsers.ParseProject(), locations_client=scheduler_api.locations)
    region_ref = parsers.ParseLocation(app_location).RelativeName()
    project = os.path.basename(str(parsers.ParseProject()))
    existing_jobs_dict = _BuildJobsMappingDict(existing_jobs, project)
    responses = []
    if cron_yaml.cron:
        for yaml_job in cron_yaml.cron:
            _ReplaceDefaultRetryParamsForYamlJob(yaml_job)
            job_key = _CreateUniqueJobKeyForYamlJob(yaml_job)
            if job_key in existing_jobs_dict and existing_jobs_dict[job_key]:
                existing_jobs_dict[job_key].pop()
                continue
            job = CreateJobInstance(scheduler_api, yaml_job)
            responses.append(jobs_client.Create(region_ref, job))
    for jobs_list in existing_jobs_dict.values():
        for yaml_job in jobs_list:
            jobs_client.Delete(yaml_job.name)
    return responses