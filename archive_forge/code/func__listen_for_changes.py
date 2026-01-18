import contextlib
import copy
import functools
import weakref
from oslo_concurrency import lockutils
from oslo_config import cfg
from oslo_db import api as oslo_db_api
from oslo_db import exception as db_exc
from oslo_db.sqlalchemy import enginefacade
from oslo_log import log as logging
from oslo_utils import excutils
from osprofiler import opts as profiler_opts
import osprofiler.sqlalchemy
from pecan import util as p_util
import sqlalchemy
from sqlalchemy import event  # noqa
from sqlalchemy import exc as sql_exc
from sqlalchemy import orm
from sqlalchemy.orm import exc
from neutron_lib._i18n import _
from neutron_lib.db import model_base
from neutron_lib import exceptions
from neutron_lib.objects import exceptions as obj_exc
@event.listens_for(model_base.BASEV2, 'attribute_instrument', propagate=True)
def _listen_for_changes(cls, key, inst):
    mapper = sqlalchemy.inspect(cls)
    if key not in mapper.relationships:
        return
    prop = inst.property
    if prop.direction is orm.interfaces.MANYTOONE:
        for col in prop.local_columns:
            colkey = mapper.get_property_by_column(col).key
            _expire_prop_on_col(cls, prop, colkey)
    elif prop.direction is orm.interfaces.ONETOMANY:
        remote_mapper = prop.mapper
        if not prop.back_populates:
            name = '_%s_backref' % prop.key
            backref_prop = orm.relationship(prop.parent, back_populates=prop.key)
            remote_mapper.add_property(name, backref_prop)
            prop.back_populates = name