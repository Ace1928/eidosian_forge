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
def _update_atom_details(self, conn, ad, e_ad):
    e_ad.merge(ad)
    conn.execute(sql.update(self._tables.atomdetails).where(self._tables.atomdetails.c.uuid == e_ad.uuid).values(e_ad.to_dict()))