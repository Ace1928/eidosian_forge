from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import itertools
import logging
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional
import uuid
from absl import flags
from google.api_core.iam import Policy
from googleapiclient import http as http_request
import inflection
from clients import bigquery_client
from clients import client_dataset
from clients import client_reservation
from clients import table_reader as bq_table_reader
from clients import utils as bq_client_utils
from utils import bq_api_utils
from utils import bq_error
from utils import bq_id_utils
from utils import bq_processor_utils
def WaitJob(self, job_reference, status='DONE', wait=sys.maxsize, wait_printer_factory: Optional[Callable[[], bq_client_utils.TransitionWaitPrinter]]=None):
    """Poll for a job to run until it reaches the requested status.

    Arguments:
      job_reference: JobReference to poll.
      status: (optional, default 'DONE') Desired job status.
      wait: (optional, default maxint) Max wait time.
      wait_printer_factory: (optional, defaults to self.wait_printer_factory)
        Returns a subclass of WaitPrinter that will be called after each job
        poll.

    Returns:
      The job object returned by the final status call.

    Raises:
      StopIteration: If polling does not reach the desired state before
        timing out.
      ValueError: If given an invalid wait value.
    """
    bq_id_utils.typecheck(job_reference, bq_id_utils.ApiClientHelper.JobReference, method='WaitJob')
    start_time = time.time()
    job = None
    if wait_printer_factory:
        printer = wait_printer_factory()
    else:
        printer = self.wait_printer_factory()
    waits = itertools.chain(itertools.repeat(1, 8), range(2, 30, 3), itertools.repeat(30))
    current_wait = 0
    current_status = 'UNKNOWN'
    in_error_state = False
    while current_wait <= wait:
        try:
            done, job = self.PollJob(job_reference, status=status, wait=wait)
            current_status = job['status']['state']
            in_error_state = False
            if done:
                printer.Print(job_reference.jobId, current_wait, current_status)
                break
        except bq_error.BigqueryCommunicationError as e:
            logging.warning('Transient error during job status check: %s', e)
        except bq_error.BigqueryBackendError as e:
            logging.warning('Transient error during job status check: %s', e)
        except BigqueryServiceError as e:
            if in_error_state:
                raise
            in_error_state = True
        for _ in range(next(waits)):
            current_wait = time.time() - start_time
            printer.Print(job_reference.jobId, current_wait, current_status)
            time.sleep(1)
    else:
        raise StopIteration('Wait timed out. Operation not finished, in state %s' % (current_status,))
    printer.Done()
    return job