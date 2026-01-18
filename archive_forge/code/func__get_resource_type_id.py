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
def _get_resource_type_id(meta, conn, name):
    rt_table = get_metadef_resource_types_table(meta, conn)
    with conn.begin():
        resource_type = conn.execute(select(rt_table.c.id).where(rt_table.c.name == name).select_from(rt_table)).fetchone()
        if resource_type:
            return resource_type[0]
    return None