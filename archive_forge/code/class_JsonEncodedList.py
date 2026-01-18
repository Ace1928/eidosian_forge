import json
from sqlalchemy.dialects import mysql
from sqlalchemy.types import Integer, Text, TypeDecorator
class JsonEncodedList(JsonEncodedType):
    """Represents list serialized as json-encoded string in db.

    Note that this type does NOT track mutations. If you want to update it, you
    have to assign existing value to a temporary variable, update, then assign
    back. See this page for more robust work around:
    http://docs.sqlalchemy.org/en/rel_1_0/orm/extensions/mutable.html
    """
    type = list
    cache_ok = True
    'This type is safe to cache.'