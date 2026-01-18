from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import datetime
import os
import sys
import textwrap
import time
from typing import Optional, TextIO
from absl import app
from absl import flags
import termcolor
import bq_flags
import bq_utils
from clients import utils as bq_client_utils
from frontend import bigquery_command
from frontend import bq_cached_client
from frontend import utils as frontend_utils
from utils import bq_error
from utils import bq_id_utils
from utils import bq_logging
from utils import bq_processor_utils
from pyglib import stringutil
class Cancel(bigquery_command.BigqueryCmd):
    """Attempt to cancel the specified job if it is running."""
    usage = 'cancel [--nosync] [<job_id>]'

    def __init__(self, name: str, fv: flags.FlagValues):
        super(Cancel, self).__init__(name, fv)
        self._ProcessCommandRc(fv)

    def RunWithArgs(self, job_id: str='') -> Optional[int]:
        """Request a cancel and waits for the job to be cancelled.

    Requests a cancel and then either:
    a) waits until the job is done if the sync flag is set [default], or
    b) returns immediately if the sync flag is not set.
    Not all job types support a cancel, an error is returned if it cannot be
    cancelled. Even for jobs that support a cancel, success is not guaranteed,
    the job may have completed by the time the cancel request is noticed, or
    the job may be in a stage where it cannot be cancelled.

    Examples:
      bq cancel job_id  # Requests a cancel and waits until the job is done.
      bq --nosync cancel job_id  # Requests a cancel and returns immediately.

    Arguments:
      job_id: Job ID to cancel.
    """
        client = bq_cached_client.Client.Get()
        job_reference_dict = dict(client.GetJobReference(job_id, FLAGS.location))
        job = client.CancelJob(job_id=job_reference_dict['jobId'], location=job_reference_dict['location'])
        frontend_utils.PrintObjectInfo(job, bq_id_utils.ApiClientHelper.JobReference.Create(**job['jobReference']), custom_format='show')
        status = job['status']
        if status['state'] == 'DONE':
            if 'errorResult' in status and 'reason' in status['errorResult'] and (status['errorResult']['reason'] == 'stopped'):
                print('Job has been cancelled successfully.')
            else:
                print('Job completed before it could be cancelled.')
        else:
            print('Job cancel has been requested.')
        return 0