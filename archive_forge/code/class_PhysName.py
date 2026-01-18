import random
import string
import uuid
import warnings
import fixtures
from oslo_config import cfg
from oslo_db import options
from oslo_serialization import jsonutils
import sqlalchemy
from sqlalchemy import exc as sqla_exc
from heat.common import context
from heat.db import api as db_api
from heat.db import models
from heat.engine import environment
from heat.engine import node_data
from heat.engine import resource
from heat.engine import stack
from heat.engine import template
class PhysName(object):
    mock_short_id = 'x' * 12

    def __init__(self, stack_name, resource_name, limit=255):
        name = '%s-%s-%s' % (stack_name, resource_name, self.mock_short_id)
        self._physname = resource.Resource.reduce_physical_resource_name(name, limit)
        self.stack, self.res, self.sid = self._physname.rsplit('-', 2)

    def __eq__(self, physical_name):
        try:
            stk, res, short_id = str(physical_name).rsplit('-', 2)
        except ValueError:
            return False
        if len(short_id) != len(self.mock_short_id):
            return False
        if not isinstance(self.stack, PhysName) and 3 < len(stk) < len(self.stack):
            our_stk = self.stack[:2] + '-' + self.stack[3 - len(stk):]
        else:
            our_stk = self.stack
        return stk == our_stk and res == self.res

    def __hash__(self):
        return hash(self.stack) ^ hash(self.res)

    def __ne__(self, physical_name):
        return not self.__eq__(physical_name)

    def __repr__(self):
        return self._physname