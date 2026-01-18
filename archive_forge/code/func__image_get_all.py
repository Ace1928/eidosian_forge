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
def _image_get_all(context, session, filters=None, marker=None, limit=None, sort_key=None, sort_dir=None, member_status='accepted', is_public=None, admin_as_user=False, return_tag=False, v1_mode=False):
    sort_key = ['created_at'] if not sort_key else sort_key
    default_sort_dir = 'desc'
    if not sort_dir:
        sort_dir = [default_sort_dir] * len(sort_key)
    elif len(sort_dir) == 1:
        default_sort_dir = sort_dir[0]
        sort_dir *= len(sort_key)
    filters = filters or {}
    visibility = filters.pop('visibility', None)
    showing_deleted = 'changes-since' in filters or filters.get('deleted', False)
    img_cond, prop_cond, tag_cond = _make_conditions_from_filters(filters, is_public)
    query = _select_images_query(context, session, img_cond, admin_as_user, member_status, visibility)
    if visibility is not None:
        if visibility != 'all':
            query = query.filter(models.Image.visibility == visibility)
    elif context.owner is None:
        query = query.filter(models.Image.visibility != 'community')
    else:
        community_filters = [models.Image.owner == context.owner, models.Image.visibility != 'community']
        query = query.filter(sa_sql.or_(*community_filters))
    if prop_cond:
        for alias, prop_condition in prop_cond:
            query = query.join(alias).filter(sa_sql.and_(*prop_condition))
    if tag_cond:
        for alias, tag_condition in tag_cond:
            query = query.join(alias).filter(sa_sql.and_(*tag_condition))
    marker_image = None
    if marker is not None:
        marker_image = _image_get(context, session, marker, force_show_deleted=showing_deleted)
    for key in ['created_at', 'id']:
        if key not in sort_key:
            sort_key.append(key)
            sort_dir.append(default_sort_dir)
    query = _paginate_query(query, models.Image, limit, sort_key, marker=marker_image, sort_dir=None, sort_dirs=sort_dir)
    query = query.options(sa_orm.joinedload(models.Image.properties)).options(sa_orm.joinedload(models.Image.locations))
    if return_tag:
        query = query.options(sa_orm.joinedload(models.Image.tags))
    images = []
    for image in query.all():
        image_dict = image.to_dict()
        image_dict = _normalize_locations(context, image_dict, force_show_deleted=showing_deleted)
        if return_tag:
            image_dict = _normalize_tags(image_dict)
        if v1_mode:
            image_dict = db_utils.mutate_image_dict_to_v1(image_dict)
        images.append(image_dict)
    return images