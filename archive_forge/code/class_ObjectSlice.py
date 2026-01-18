import json
import logging
import uuid
from peewee import *
from peewee import ColumnBase
from peewee import Expression
from peewee import Node
from peewee import NodeList
from peewee import __deprecated__
class ObjectSlice(_LookupNode):

    @classmethod
    def create(cls, node, value):
        if isinstance(value, slice):
            parts = [value.start or 0, value.stop or 0]
        elif isinstance(value, int):
            parts = [value]
        elif isinstance(value, Node):
            parts = value
        else:
            parts = [int(i) for i in value.split(':')]
        return cls(node, parts)

    def __sql__(self, ctx):
        ctx.sql(self.node)
        if isinstance(self.parts, Node):
            ctx.literal('[').sql(self.parts).literal(']')
        else:
            ctx.literal('[%s]' % ':'.join((str(p + 1) for p in self.parts)))
        return ctx

    def __getitem__(self, value):
        return ObjectSlice.create(self, value)