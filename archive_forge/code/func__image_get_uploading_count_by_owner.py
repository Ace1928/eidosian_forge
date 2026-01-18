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
def _image_get_uploading_count_by_owner(context, session, owner):
    """Return a count of the images uploading or importing."""
    importing_statuses = ('saving', 'uploading', 'importing')
    query = session.query(models.Image)
    query = query.filter(models.Image.owner == owner)
    query = query.filter(models.Image.status.in_(importing_statuses))
    uploading = query.count()
    props = session.query(models.ImageProperty).filter(models.ImageProperty.name == 'os_glance_importing_to_stores', models.ImageProperty.value != '').subquery()
    query = session.query(models.Image)
    query = query.join(props, props.c.image_id == models.Image.id)
    query = query.filter(models.Image.owner == owner)
    query = query.filter(~models.Image.status.in_(importing_statuses + ('killed', 'deleted')))
    copying = query.count()
    return uploading + copying