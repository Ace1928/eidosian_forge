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
def _insert_flow_details(self, conn, fd, parent_uuid):
    value = fd.to_dict()
    value['parent_uuid'] = parent_uuid
    conn.execute(sql.insert(self._tables.flowdetails).values(**value))
    for ad in fd:
        self._insert_atom_details(conn, ad, fd.uuid)