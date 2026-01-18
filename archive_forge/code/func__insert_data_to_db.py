import json
import os
from os.path import isfile
from os.path import join
import re
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
import sqlalchemy
from sqlalchemy import and_
from sqlalchemy.schema import MetaData
from sqlalchemy.sql import select
from glance.common import timeutils
from glance.i18n import _, _LE, _LI, _LW
def _insert_data_to_db(conn, table, values, log_exception=True):
    try:
        with conn.begin():
            conn.execute(table.insert().values(values))
    except sqlalchemy.exc.IntegrityError:
        if log_exception:
            LOG.warning(_LW('Duplicate entry for values: %s'), values)