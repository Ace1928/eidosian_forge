import collections
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from oslo_utils import uuidutils
from sqlalchemy import Table, Column, String, ForeignKey, DateTime, Enum
from sqlalchemy_utils.types import json as json_type
from taskflow.persistence import models
from taskflow import states
class JSONType(json_type.JSONType):
    """Customized JSONType using oslo.serialization for json operations"""

    def process_bind_param(self, value, dialect):
        if dialect.name == 'postgresql' and json_type.has_postgres_json:
            return value
        if value is not None:
            value = jsonutils.dumps(value)
        return value

    def process_result_value(self, value, dialect):
        if dialect.name == 'postgresql':
            return value
        if value is not None:
            value = jsonutils.loads(value)
        return value