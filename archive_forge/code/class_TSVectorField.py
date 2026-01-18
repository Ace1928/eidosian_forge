import json
import logging
import uuid
from peewee import *
from peewee import ColumnBase
from peewee import Expression
from peewee import Node
from peewee import NodeList
from peewee import __deprecated__
class TSVectorField(IndexedFieldMixin, TextField):
    field_type = 'TSVECTOR'
    __hash__ = Field.__hash__

    def match(self, query, language=None, plain=False):
        params = (language, query) if language is not None else (query,)
        func = fn.plainto_tsquery if plain else fn.to_tsquery
        return Expression(self, TS_MATCH, func(*params))