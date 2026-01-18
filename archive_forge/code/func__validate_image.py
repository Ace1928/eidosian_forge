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
def _validate_image(values, mandatory_status=True):
    """
    Validates the incoming data and raises a Invalid exception
    if anything is out of order.

    :param values: Mapping of image metadata to check
    :param mandatory_status: Whether to validate status from values
    """
    if mandatory_status:
        status = values.get('status')
        if not status:
            msg = 'Image status is required.'
            raise exception.Invalid(msg)
        if status not in STATUSES:
            msg = "Invalid image status '%s' for image." % status
            raise exception.Invalid(msg)
    _validate_db_int(min_disk=values.get('min_disk'), min_ram=values.get('min_ram'))
    return values