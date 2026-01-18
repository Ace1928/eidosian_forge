import datetime
import itertools
import threading
from oslo_config import cfg
from oslo_db import exception as db_exception
from oslo_db.sqlalchemy import session as oslo_db_session
from oslo_log import log as logging
from oslo_utils import excutils
import osprofiler.sqlalchemy
from retrying import retry
import sqlalchemy
from sqlalchemy.ext.compiler import compiles
from sqlalchemy import MetaData, Table
import sqlalchemy.orm as sa_orm
from sqlalchemy import sql
import sqlalchemy.sql as sa_sql
from glance.common import exception
from glance.common import timeutils
from glance.common import utils
from glance.db.sqlalchemy.metadef_api import (resource_type
from glance.db.sqlalchemy.metadef_api import (resource_type_association
from glance.db.sqlalchemy.metadef_api import namespace as metadef_namespace_api
from glance.db.sqlalchemy.metadef_api import object as metadef_object_api
from glance.db.sqlalchemy.metadef_api import property as metadef_property_api
from glance.db.sqlalchemy.metadef_api import tag as metadef_tag_api
from glance.db.sqlalchemy import models
from glance.db import utils as db_utils
from glance.i18n import _, _LW, _LI, _LE
def _image_get_disk_usage_by_owner(context, session, owner, image_id=None):
    query = session.query(models.Image)
    query = query.filter(models.Image.owner == owner)
    if image_id is not None:
        query = query.filter(models.Image.id != image_id)
    query = query.filter(models.Image.size > 0)
    query = query.filter(~models.Image.status.in_(['killed', 'deleted']))
    images = query.all()
    total = 0
    for i in images:
        locations = [location for location in i.locations if location['status'] != 'deleted']
        total += i.size * len(locations)
    return total