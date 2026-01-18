import copy
import functools
import threading
import time
from oslo_utils import strutils
import sqlalchemy as sa
from sqlalchemy import exc as sa_exc
from sqlalchemy import pool as sa_pool
from sqlalchemy import sql
import tenacity
from taskflow import exceptions as exc
from taskflow import logging
from taskflow.persistence.backends.sqlalchemy import migration
from taskflow.persistence.backends.sqlalchemy import tables
from taskflow.persistence import base
from taskflow.persistence import models
from taskflow.utils import eventlet_utils
from taskflow.utils import misc
@tenacity.retry(stop=tenacity.stop_after_attempt(max(0, int(max_retries))), wait=tenacity.wait_exponential(), reraise=True, retry=tenacity.retry_if_exception(_retry_on_exception))
def _try_connect(engine):
    with engine.connect():
        pass