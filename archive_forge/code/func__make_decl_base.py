import gc
from sqlalchemy.ext import declarative
from sqlalchemy import orm
import testtools
from neutron_lib.db import standard_attr
from neutron_lib.tests import _base as base
def _make_decl_base(self):
    try:

        class BaseV2(orm.DeclarativeBase, standard_attr.model_base.NeutronBaseV2):
            pass
        return BaseV2
    except AttributeError:
        return declarative.declarative_base(cls=standard_attr.model_base.NeutronBaseV2)