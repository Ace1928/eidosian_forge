import json
import logging
import uuid
from peewee import *
from peewee import ColumnBase
from peewee import Expression
from peewee import Node
from peewee import NodeList
from peewee import __deprecated__
class JsonLookup(_JsonLookupBase):

    def __getitem__(self, value):
        return JsonLookup(self.node, self.parts + [value], self._as_json)

    def __sql__(self, ctx):
        ctx.sql(self.node)
        for part in self.parts[:-1]:
            ctx.literal('->').sql(part)
        if self.parts:
            ctx.literal('->' if self._as_json else '->>').sql(self.parts[-1])
        return ctx