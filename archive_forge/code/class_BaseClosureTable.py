import json
import math
import re
import struct
import sys
from peewee import *
from peewee import ColumnBase
from peewee import EnclosedNodeList
from peewee import Entity
from peewee import Expression
from peewee import Insert
from peewee import Node
from peewee import NodeList
from peewee import OP
from peewee import VirtualField
from peewee import merge_dict
from peewee import sqlite3
class BaseClosureTable(VirtualModel):
    depth = VirtualField(IntegerField)
    id = VirtualField(IntegerField)
    idcolumn = VirtualField(TextField)
    parentcolumn = VirtualField(TextField)
    root = VirtualField(IntegerField)
    tablename = VirtualField(TextField)

    class Meta:
        extension_module = 'transitive_closure'

    @classmethod
    def descendants(cls, node, depth=None, include_node=False):
        query = model_class.select(model_class, cls.depth.alias('depth')).join(cls, on=source_key == cls.id).where(cls.root == node).objects()
        if depth is not None:
            query = query.where(cls.depth == depth)
        elif not include_node:
            query = query.where(cls.depth > 0)
        return query

    @classmethod
    def ancestors(cls, node, depth=None, include_node=False):
        query = model_class.select(model_class, cls.depth.alias('depth')).join(cls, on=source_key == cls.root).where(cls.id == node).objects()
        if depth:
            query = query.where(cls.depth == depth)
        elif not include_node:
            query = query.where(cls.depth > 0)
        return query

    @classmethod
    def siblings(cls, node, include_node=False):
        if referencing_class is model_class:
            fk_value = node.__data__.get(foreign_key.name)
            query = model_class.select().where(foreign_key == fk_value)
        else:
            siblings = referencing_class.select(referencing_key).join(cls, on=foreign_key == cls.root).where((cls.id == node) & (cls.depth == 1))
            query = model_class.select().where(source_key << siblings).objects()
        if not include_node:
            query = query.where(source_key != node)
        return query