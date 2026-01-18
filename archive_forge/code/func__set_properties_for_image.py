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
@utils.no_4byte_params
def _set_properties_for_image(context, session, image_ref, properties, purge_props=False, atomic_props=None):
    """
    Create or update a set of image_properties for a given image

    :param context: Request context
    :param session: A SQLAlchemy session to use
    :param image_ref: An Image object
    :param properties: A dict of properties to set
    :param purge_props: If True, delete properties in the database
                        that are not in properties
    :param atomic_props: If non-None, skip update/create/delete of properties
                         named in this list
    """
    if atomic_props is None:
        atomic_props = []
    orig_properties = {}
    for prop_ref in image_ref.properties:
        orig_properties[prop_ref.name] = prop_ref
    for name, value in properties.items():
        prop_values = {'image_id': image_ref.id, 'name': name, 'value': value}
        if name in atomic_props:
            continue
        elif name in orig_properties:
            prop_ref = orig_properties[name]
            _image_property_update(context, session, prop_ref, prop_values)
        else:
            _image_property_create(context, session, prop_values)
    if purge_props:
        for key in orig_properties.keys():
            if key in atomic_props:
                continue
            elif key not in properties:
                prop_ref = orig_properties[key]
                _image_property_delete(context, session, prop_ref.name, image_ref.id)