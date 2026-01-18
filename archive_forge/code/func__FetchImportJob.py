from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import sys
from googlecloudsdk.api_lib.cloudkms import base as cloudkms_base
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.kms import flags
from googlecloudsdk.command_lib.kms import maps
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
def _FetchImportJob(self, args, import_job_name, client, messages):
    import_job = client.projects_locations_keyRings_importJobs.Get(messages.CloudkmsProjectsLocationsKeyRingsImportJobsGetRequest(name=import_job_name))
    if import_job.state != messages.ImportJob.StateValueValuesEnum.ACTIVE:
        raise exceptions.BadArgumentException('import-job', 'Import job [{0}] is not active (state is {1}).'.format(import_job_name, import_job.state))
    return import_job