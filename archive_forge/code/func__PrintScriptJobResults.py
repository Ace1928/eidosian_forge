from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import logging
import sys
from typing import Optional
from absl import app
from absl import flags
from pyglib import appcommands
from clients import bigquery_client
from clients import bigquery_client_extended
from clients import utils as bq_client_utils
from frontend import bigquery_command
from frontend import bq_cached_client
from frontend import utils as frontend_utils
from frontend import utils_data_transfer
from utils import bq_error
from utils import bq_id_utils
from pyglib import stringutil
def _PrintScriptJobResults(self, client: bigquery_client_extended.BigqueryClientExtended, job) -> None:
    """Prints the results of a successful script job.

    This function is invoked only for successful script jobs.  Prints the output
    of each successful child job representing a statement to stdout.

    Child jobs representing expression evaluations are not printed, as are child
    jobs which failed, but whose error was handled elsewhere in the script.

    Depending on flags, the output is printed in either free-form or
    json style.

    Args:
      client: Bigquery client object
      job: json of the script job, expressed as a dictionary
    """
    child_jobs = list(client.ListJobs(reference=bq_id_utils.ApiClientHelper.ProjectReference.Create(projectId=job['jobReference']['projectId']), max_results=self.max_child_jobs + 1, all_users=False, min_creation_time=None, max_creation_time=None, page_token=None, parent_job_id=job['jobReference']['jobId']))
    if not child_jobs:
        self.PrintNonScriptQueryJobResults(client, job)
        return
    child_jobs.sort(key=lambda job: job['statistics']['creationTime'])
    if len(child_jobs) == self.max_child_jobs + 1:
        sys.stderr.write('Showing only the final result because the number of child jobs exceeds --max_child_jobs (%s).\n' % self.max_child_jobs)
        self.PrintNonScriptQueryJobResults(client, job)
        return
    statement_child_jobs = [job for job in child_jobs if job.get('statistics', {}).get('scriptStatistics', {}).get('evaluationKind', '') == 'STATEMENT']
    is_raw_json = FLAGS.format == 'json'
    is_json = is_raw_json or FLAGS.format == 'prettyjson'
    if is_json:
        sys.stdout.write('[')
    statements_printed = 0
    for i, child_job_info in enumerate(statement_child_jobs):
        if bq_client_utils.IsFailedJob(child_job_info):
            continue
        if statements_printed >= self.max_statement_results:
            if not is_json:
                sys.stdout.write('Maximum statement results limit reached. Specify --max_statement_results to increase this limit.\n')
            break
        if is_json:
            if i > 0:
                if is_raw_json:
                    sys.stdout.write(',')
                else:
                    sys.stdout.write(',\n')
        else:
            stack_frames = child_job_info.get('statistics', {}).get('scriptStatistics', {}).get('stackFrames', [])
            if len(stack_frames) <= 0:
                break
            sys.stdout.write('%s; ' % stringutil.ensure_str(stack_frames[0].get('text', '')))
            if len(stack_frames) >= 2:
                sys.stdout.write('\n')
            for stack_frame in stack_frames:
                sys.stdout.write('-- at %s[%d:%d]\n' % (stack_frame.get('procedureId', ''), stack_frame['startLine'], stack_frame['startColumn']))
        self.PrintNonScriptQueryJobResults(client, child_job_info)
        statements_printed = statements_printed + 1
    if is_json:
        sys.stdout.write(']\n')