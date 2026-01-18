from __future__ import annotations
import typing as t
from sqlglot.dialects.dialect import DialectType
from sqlglot.helper import dict_depth
from sqlglot.schema import AbstractMappingSchema, normalize_name
class RangeReader:

    def __init__(self, table):
        self.table = table
        self.range = range(0)

    def __len__(self):
        return len(self.range)

    def __getitem__(self, column):
        return (self.table[i][column] for i in self.range)