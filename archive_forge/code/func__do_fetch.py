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
def _do_fetch(p):
    p.multi()
    p.hexists(listings_key, listings_sub_key)
    p.exists(owner_key)
    job_exists, owner_exists = p.execute()
    if not job_exists:
        if owner_exists:
            LOG.info("Unexpected owner key found at '%s' when job key '%s[%s]' was not found", owner_key, listings_key, listings_sub_key)
        return states.COMPLETE
    elif owner_exists:
        return states.CLAIMED
    else:
        return states.UNCLAIMED