from __future__ import annotations
from typing import Any, MutableSequence
from rdflib.graph import ConjunctiveGraph, Graph
from rdflib.parser import InputSource, Parser
from .notation3 import RDFSink, SinkParser
class TrigSinkParser(SinkParser):

    def directiveOrStatement(self, argstr: str, h: int) -> int:
        i = self.skipSpace(argstr, h)
        if i < 0:
            return i
        j = self.graph(argstr, i)
        if j >= 0:
            return j
        j = self.sparqlDirective(argstr, i)
        if j >= 0:
            return j
        j = self.directive(argstr, i)
        if j >= 0:
            return self.checkDot(argstr, j)
        j = self.statement(argstr, i)
        if j >= 0:
            return self.checkDot(argstr, j)
        return j

    def labelOrSubject(self, argstr: str, i: int, res: MutableSequence[Any]) -> int:
        j = self.skipSpace(argstr, i)
        if j < 0:
            return j
        i = j
        j = self.uri_ref2(argstr, i, res)
        if j >= 0:
            return j
        if argstr[i] == '[':
            j = self.skipSpace(argstr, i + 1)
            if j < 0:
                self.BadSyntax(argstr, i, 'Expected ] got EOF')
            if argstr[j] == ']':
                res.append(self.blankNode())
                return j + 1
        return -1

    def graph(self, argstr: str, i: int) -> int:
        """
        Parse trig graph, i.e.

           <urn:graphname> = { .. triples .. }

        return -1 if it doesn't look like a graph-decl
        raise Exception if it looks like a graph, but isn't.
        """
        need_graphid = False
        j = self.sparqlTok('GRAPH', argstr, i)
        if j >= 0:
            i = j
            need_graphid = True
        r: MutableSequence[Any] = []
        j = self.labelOrSubject(argstr, i, r)
        if j >= 0:
            graph = r[0]
            i = j
        elif need_graphid:
            self.BadSyntax(argstr, i, 'GRAPH keyword must be followed by graph name')
        else:
            graph = self._store.graph.identifier
        j = self.skipSpace(argstr, i)
        if j < 0:
            self.BadSyntax(argstr, i, 'EOF found when expected graph')
        if argstr[j:j + 1] == '=':
            i = self.skipSpace(argstr, j + 1)
            if i < 0:
                self.BadSyntax(argstr, i, "EOF found when expecting '{'")
        else:
            i = j
        if argstr[i:i + 1] != '{':
            return -1
        j = i + 1
        if self._context is not None:
            self.BadSyntax(argstr, i, 'Nested graphs are not allowed')
        oldParentContext = self._parentContext
        self._parentContext = self._context
        reason2 = self._reason2
        self._reason2 = becauseSubGraph
        self._context = self._store.newGraph(graph)
        while 1:
            i = self.skipSpace(argstr, j)
            if i < 0:
                self.BadSyntax(argstr, i, "needed '}', found end.")
            if argstr[i:i + 1] == '}':
                j = i + 1
                break
            j = self.directiveOrStatement(argstr, i)
            if j < 0:
                self.BadSyntax(argstr, i, "expected statement or '}'")
        self._context = self._parentContext
        self._reason2 = reason2
        self._parentContext = oldParentContext
        return j