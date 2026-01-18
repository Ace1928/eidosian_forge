from __future__ import annotations
import collections
import datetime
import itertools
import typing as t
from collections.abc import Mapping, MutableMapping
from typing import (
import isodate
import rdflib.plugins.sparql
from rdflib.graph import ConjunctiveGraph, Graph
from rdflib.namespace import NamespaceManager
from rdflib.plugins.sparql.parserutils import CompValue
from rdflib.term import BNode, Identifier, Literal, Node, URIRef, Variable
class FrozenBindings(FrozenDict):

    def __init__(self, ctx: 'QueryContext', *args, **kwargs):
        FrozenDict.__init__(self, *args, **kwargs)
        self.ctx = ctx

    def __getitem__(self, key: Union[Identifier, str]) -> Identifier:
        if not isinstance(key, Node):
            key = Variable(key)
        if not isinstance(key, (BNode, Variable)):
            return key
        if key not in self._d:
            return self.ctx.initBindings[key]
        else:
            return self._d[key]

    def project(self, vars: Container[Variable]) -> 'FrozenBindings':
        return FrozenBindings(self.ctx, (x for x in self.items() if x[0] in vars))

    def merge(self, other: t.Mapping[Identifier, Identifier]) -> 'FrozenBindings':
        res = FrozenBindings(self.ctx, itertools.chain(self.items(), other.items()))
        return res

    @property
    def now(self) -> datetime.datetime:
        return self.ctx.now

    @property
    def bnodes(self) -> t.Mapping[Identifier, BNode]:
        return self.ctx.bnodes

    @property
    def prologue(self) -> Optional['Prologue']:
        return self.ctx.prologue

    def forget(self, before: 'QueryContext', _except: Optional[Container[Variable]]=None) -> FrozenBindings:
        """
        return a frozen dict only of bindings made in self
        since before
        """
        if not _except:
            _except = []
        return FrozenBindings(self.ctx, (x for x in self.items() if x[0] in _except or x[0] in self.ctx.initBindings or before[x[0]] is None))

    def remember(self, these) -> FrozenBindings:
        """
        return a frozen dict only of bindings in these
        """
        return FrozenBindings(self.ctx, (x for x in self.items() if x[0] in these))