from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import datetime
import functools
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple
from absl import app
from absl import flags
import yaml
import table_formatter
import bq_utils
from clients import utils as bq_client_utils
from utils import bq_error
from utils import bq_id_utils
from pyglib import stringutil
def PrintJobMessages(printable_job_info):
    """Prints additional info from a job formatted for printing.

  If the job had a fatal error, non-fatal warnings are not shown.

  If any error/warning does not have a 'message' key, printable_job_info must
  have 'jobReference' identifying the job.

  For DML queries prints number of affected rows.
  For DDL queries prints the performed operation and the target.
  """
    job_ref = '(unknown)'
    if 'jobReference' in printable_job_info:
        job_ref = printable_job_info['jobReference']
    if printable_job_info['State'] == 'FAILURE':
        error_result = printable_job_info['status']['errorResult']
        error_ls = printable_job_info['status'].get('errors', [])
        error = bq_error.CreateBigqueryError(error_result, error_result, error_ls)
        print('Error encountered during job execution:\n%s\n' % (error,))
    elif 'errors' in printable_job_info['status']:
        warnings = printable_job_info['status']['errors']
        print('Warning%s encountered during job execution:\n' % ('' if len(warnings) == 1 else 's'))
        recommend_show = False
        for w in warnings:
            if 'message' not in w:
                recommend_show = True
            else:
                if 'location' in w:
                    message = '[%s] %s' % (w['location'], w['message'])
                else:
                    message = w['message']
                if message is not None:
                    message = message.encode('utf-8')
                print('%s\n' % message)
        if recommend_show:
            print('Use "bq show -j %s" to view job warnings.' % job_ref)
    elif 'Affected Rows' in printable_job_info:
        print('Number of affected rows: %s\n' % printable_job_info['Affected Rows'])
    elif 'DDL Target Table' in printable_job_info:
        ddl_target_table = printable_job_info['DDL Target Table']
        project_id = ddl_target_table.get('projectId')
        dataset_id = ddl_target_table.get('datasetId')
        table_id = ddl_target_table.get('tableId')
        op = _DDL_OPERATION_MAP.get(printable_job_info.get('DDL Operation Performed'))
        if project_id and dataset_id and table_id and op:
            if 'DDL Affected Row Access Policy Count' in printable_job_info:
                ddl_affected_row_access_policy_count = printable_job_info['DDL Affected Row Access Policy Count']
                print('{op} {count} row access policies on table {project}.{dataset}.{table}\n'.format(op=op, count=ddl_affected_row_access_policy_count, project=project_id, dataset=dataset_id, table=table_id))
            elif 'Statement Type' in printable_job_info and 'INDEX' in printable_job_info['Statement Type']:
                if 'SEARCH_INDEX' in printable_job_info['Statement Type']:
                    print('%s search index on table %s.%s.%s\n' % (stringutil.ensure_str(op), stringutil.ensure_str(project_id), stringutil.ensure_str(dataset_id), stringutil.ensure_str(table_id)))
                elif 'VECTOR_INDEX' in printable_job_info['Statement Type']:
                    index_progress_instruction = ''
                    if printable_job_info.get('DDL Operation Performed') in ('CREATE', 'REPLACE'):
                        index_progress_instruction = 'Please use INFORMATION_SCHEMA to check the progress of the index.\n'
                    print('%s vector index on table %s.%s.%s\n%s' % (stringutil.ensure_str(op), stringutil.ensure_str(project_id), stringutil.ensure_str(dataset_id), stringutil.ensure_str(table_id), stringutil.ensure_str(index_progress_instruction)))
            else:
                print('%s %s.%s.%s\n' % (stringutil.ensure_str(op), stringutil.ensure_str(project_id), stringutil.ensure_str(dataset_id), stringutil.ensure_str(table_id)))
    elif 'DDL Target Routine' in printable_job_info:
        ddl_target_routine = printable_job_info['DDL Target Routine']
        project_id = ddl_target_routine.get('projectId')
        dataset_id = ddl_target_routine.get('datasetId')
        routine_id = ddl_target_routine.get('routineId')
        op = _DDL_OPERATION_MAP.get(printable_job_info.get('DDL Operation Performed'))
        temp_object_name = MaybeGetSessionTempObjectName(dataset_id, routine_id)
        if temp_object_name is not None:
            print('%s temporary routine %s' % (op, temp_object_name))
        else:
            print('%s %s.%s.%s' % (op, project_id, dataset_id, routine_id))
    elif 'DDL Target Row Access Policy' in printable_job_info:
        ddl_target_row_access_policy = printable_job_info['DDL Target Row Access Policy']
        project_id = ddl_target_row_access_policy.get('projectId')
        dataset_id = ddl_target_row_access_policy.get('datasetId')
        table_id = ddl_target_row_access_policy.get('tableId')
        row_access_policy_id = ddl_target_row_access_policy.get('policyId')
        op = _DDL_OPERATION_MAP.get(printable_job_info.get('DDL Operation Performed'))
        if project_id and dataset_id and table_id and row_access_policy_id and op:
            print('{op} row access policy {policy} on table {project}.{dataset}.{table}'.format(op=op, policy=row_access_policy_id, project=project_id, dataset=dataset_id, table=table_id))
    elif 'Assertion' in printable_job_info:
        print('Assertion successful')
    if 'Session Id' in printable_job_info:
        print('In session: %s' % printable_job_info['Session Id'])