import abc
import os
import threading
import fasteners
from taskflow import engines
from taskflow import exceptions as excp
from taskflow.types import entity
from taskflow.types import notifier
from taskflow.utils import misc
def _flow_detail_from_job(self, job):
    """Extracts a flow detail from a job (via some manner).

        The current mechanism to accomplish this is the following choices:

        * If the job details provide a 'flow_uuid' key attempt to load this
          key from the jobs book and use that as the flow_detail to run.
        * If the job details does not have have a 'flow_uuid' key then attempt
          to examine the size of the book and if it's only one element in the
          book (aka one flow_detail) then just use that.
        * Otherwise if there is no 'flow_uuid' defined or there are > 1
          flow_details in the book raise an error that corresponds to being
          unable to locate the correct flow_detail to run.
        """
    book = job.book
    if book is None:
        raise excp.NotFound('No book found in job')
    if job.details and 'flow_uuid' in job.details:
        flow_uuid = job.details['flow_uuid']
        flow_detail = book.find(flow_uuid)
        if flow_detail is None:
            raise excp.NotFound('No matching flow detail found in jobs book for flow detail with uuid %s' % flow_uuid)
    else:
        choices = len(book)
        if choices == 1:
            flow_detail = list(book)[0]
        elif choices == 0:
            raise excp.NotFound('No flow detail(s) found in jobs book')
        else:
            raise excp.MultipleChoices('No matching flow detail found (%s choices) in jobs book' % choices)
    return flow_detail