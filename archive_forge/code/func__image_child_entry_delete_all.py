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
def _image_child_entry_delete_all(context, session, child_model_cls, image_id, delete_time=None):
    """Deletes all the child entries for the given image id.

    Deletes all the child entries of the given child entry ORM model class
    using the parent image's id.

    The child entry ORM model class can be one of the following:
    model.ImageLocation, model.ImageProperty, model.ImageMember and
    model.ImageTag.

    :param context: Request context
    :param session: A SQLAlchemy session to use
    :param child_model_cls: the ORM model class.
    :param image_id: id of the image whose child entries are to be deleted.
    :param delete_time: datetime of deletion to be set.
                        If None, uses current datetime.

    :rtype: int
    :returns: The number of child entries got soft-deleted.
    """
    query = session.query(child_model_cls).filter_by(image_id=image_id).filter_by(deleted=False)
    delete_time = delete_time or timeutils.utcnow()
    count = query.update({'deleted': True, 'deleted_at': delete_time})
    return count