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
def __status__(flag, return_highwater=False):
    """
        Expose a sqlite3_status() call for a particular flag as a property of
        the Database object.
        """

    def getter(self):
        result = sqlite_get_status(flag)
        return result[1] if return_highwater else result
    return property(getter)