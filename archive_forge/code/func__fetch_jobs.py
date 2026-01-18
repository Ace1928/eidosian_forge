import contextlib
import datetime
import functools
import re
import string
import threading
import time
import fasteners
import msgpack
from oslo_serialization import msgpackutils
from oslo_utils import excutils
from oslo_utils import strutils
from oslo_utils import timeutils
from oslo_utils import uuidutils
from redis import exceptions as redis_exceptions
from redis import sentinel
from taskflow import exceptions as exc
from taskflow.jobs import base
from taskflow import logging
from taskflow import states
from taskflow.utils import misc
from taskflow.utils import redis_utils as ru
def _fetch_jobs(self):
    with _translate_failures():
        raw_postings = self._client.hgetall(self.listings_key)
    postings = []
    for raw_job_key, raw_posting in raw_postings.items():
        try:
            job_data = self._loads(raw_posting)
            try:
                job_priority = job_data['priority']
                job_priority = base.JobPriority.convert(job_priority)
            except KeyError:
                job_priority = base.JobPriority.NORMAL
            job_created_on = job_data['created_on']
            job_uuid = job_data['uuid']
            job_name = job_data['name']
            job_sequence_id = job_data['sequence']
            job_details = job_data.get('details', {})
        except (ValueError, TypeError, KeyError, exc.JobFailure):
            with excutils.save_and_reraise_exception():
                LOG.warning('Incorrectly formatted job data found at key: %s[%s]', self.listings_key, raw_job_key, exc_info=True)
                LOG.info('Deleting invalid job data at key: %s[%s]', self.listings_key, raw_job_key)
                self._client.hdel(self.listings_key, raw_job_key)
        else:
            postings.append(RedisJob(self, job_name, job_sequence_id, raw_job_key, uuid=job_uuid, details=job_details, created_on=job_created_on, book_data=job_data.get('book'), backend=self._persistence, priority=job_priority))
    return sorted(postings, reverse=True)