from __future__ import annotations
import logging
import pathlib
import random
from io import BytesIO
from typing import (
from urllib.parse import urlparse
from urllib.request import url2pathname
import rdflib.exceptions as exceptions
import rdflib.namespace as namespace  # noqa: F401 # This is here because it is used in a docstring.
import rdflib.plugin as plugin
import rdflib.query as query
import rdflib.util  # avoid circular dependency
from rdflib.collection import Collection
from rdflib.exceptions import ParserError
from rdflib.namespace import RDF, Namespace, NamespaceManager
from rdflib.parser import InputSource, Parser, create_input_source
from rdflib.paths import Path
from rdflib.resource import Resource
from rdflib.serializer import Serializer
from rdflib.store import Store
from rdflib.term import (
class BatchAddGraph:
    """
    Wrapper around graph that turns batches of calls to Graph's add
    (and optionally, addN) into calls to batched calls to addN`.

    :Parameters:

      - graph: The graph to wrap
      - batch_size: The maximum number of triples to buffer before passing to
        Graph's addN
      - batch_addn: If True, then even calls to `addN` will be batched according to
        batch_size

    graph: The wrapped graph
    count: The number of triples buffered since initialization or the last call to reset
    batch: The current buffer of triples

    """

    def __init__(self, graph: Graph, batch_size: int=1000, batch_addn: bool=False):
        if not batch_size or batch_size < 2:
            raise ValueError('batch_size must be a positive number')
        self.graph = graph
        self.__graph_tuple = (graph,)
        self.__batch_size = batch_size
        self.__batch_addn = batch_addn
        self.reset()

    def reset(self) -> BatchAddGraph:
        """
        Manually clear the buffered triples and reset the count to zero
        """
        self.batch: List[_QuadType] = []
        self.count = 0
        return self

    def add(self, triple_or_quad: Union['_TripleType', '_QuadType']) -> 'BatchAddGraph':
        """
        Add a triple to the buffer

        :param triple: The triple to add
        """
        if len(self.batch) >= self.__batch_size:
            self.graph.addN(self.batch)
            self.batch = []
        self.count += 1
        if len(triple_or_quad) == 3:
            self.batch.append(triple_or_quad + self.__graph_tuple)
        else:
            self.batch.append(triple_or_quad)
        return self

    def addN(self, quads: Iterable['_QuadType']) -> BatchAddGraph:
        if self.__batch_addn:
            for q in quads:
                self.add(q)
        else:
            self.graph.addN(quads)
        return self

    def __enter__(self) -> BatchAddGraph:
        self.reset()
        return self

    def __exit__(self, *exc) -> None:
        if exc[0] is None:
            self.graph.addN(self.batch)