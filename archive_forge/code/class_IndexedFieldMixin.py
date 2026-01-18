import json
import logging
import uuid
from peewee import *
from peewee import ColumnBase
from peewee import Expression
from peewee import Node
from peewee import NodeList
from peewee import __deprecated__
class IndexedFieldMixin(object):
    default_index_type = 'GIN'

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('index', True)
        super(IndexedFieldMixin, self).__init__(*args, **kwargs)