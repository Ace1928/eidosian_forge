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
def _get_objects(meta, conn, namespace_id):
    objects_table = get_metadef_objects_table(meta, conn)
    with conn.begin():
        return conn.execute(objects_table.select().where(objects_table.c.namespace_id == namespace_id)).fetchall()