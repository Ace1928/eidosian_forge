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
def _get_namespace_resource_types(meta, conn, namespace_id):
    namespace_resource_types_table = get_metadef_namespace_resource_types_table(meta, conn)
    with conn.begin():
        return conn.execute(namespace_resource_types_table.select().where(namespace_resource_types_table.c.namespace_id == namespace_id)).fetchall()