import json
import logging
import uuid
from peewee import *
from peewee import ColumnBase
from peewee import Expression
from peewee import Node
from peewee import NodeList
from peewee import __deprecated__
class _LookupNode(ColumnBase):

    def __init__(self, node, parts):
        self.node = node
        self.parts = parts
        super(_LookupNode, self).__init__()

    def clone(self):
        return type(self)(self.node, list(self.parts))

    def __hash__(self):
        return hash((self.__class__.__name__, id(self)))