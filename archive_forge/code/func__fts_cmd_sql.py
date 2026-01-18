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
def _fts_cmd_sql(cls, cmd, **extra_params):
    tbl = cls._meta.entity
    columns = [tbl]
    values = [cmd]
    for key, value in extra_params.items():
        columns.append(Entity(key))
        values.append(value)
    return NodeList((SQL('INSERT INTO'), cls._meta.entity, EnclosedNodeList(columns), SQL('VALUES'), EnclosedNodeList(values)))