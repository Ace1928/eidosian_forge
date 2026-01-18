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
class JsonRepr(object):
    """Comparison class used to check the deserialisation of a JSON string.

    If a dict is dumped to json, the order is undecided, so load the string
    back to an object for comparison.
    """

    def __init__(self, data):
        """Initialise with the unserialised data."""
        self._data = data

    def __eq__(self, json_data):
        return self._data == jsonutils.loads(json_data)

    def __ne__(self, json_data):
        return not self.__eq__(json_data)

    def __repr__(self):
        return jsonutils.dumps(self._data)