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
@classmethod
def VocabModel(cls, table_type='row', table=None):
    if table_type not in ('row', 'col', 'instance'):
        raise ValueError('table_type must be either "row", "col" or "instance".')
    attr = '_vocab_model_%s' % table_type
    if not hasattr(cls, attr):

        class Meta:
            database = cls._meta.database
            table_name = table or cls._meta.table_name + '_v'
            extension_module = fn.fts5vocab(cls._meta.entity, SQL(table_type))
        attrs = {'term': VirtualField(TextField), 'doc': IntegerField(), 'cnt': IntegerField(), 'rowid': RowIDField(), 'Meta': Meta}
        if table_type == 'col':
            attrs['col'] = VirtualField(TextField)
        elif table_type == 'instance':
            attrs['offset'] = VirtualField(IntegerField)
        class_name = '%sVocab' % cls.__name__
        setattr(cls, attr, type(class_name, (VirtualModel,), attrs))
    return getattr(cls, attr)