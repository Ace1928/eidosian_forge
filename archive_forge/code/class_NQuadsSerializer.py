import warnings
from typing import IO, Optional
from rdflib.graph import ConjunctiveGraph, Graph
from rdflib.plugins.serializers.nt import _quoteLiteral
from rdflib.serializer import Serializer
from rdflib.term import Literal
class NQuadsSerializer(Serializer):

    def __init__(self, store: Graph):
        if not store.context_aware:
            raise Exception('NQuads serialization only makes sense for context-aware stores!')
        super(NQuadsSerializer, self).__init__(store)
        self.store: ConjunctiveGraph

    def serialize(self, stream: IO[bytes], base: Optional[str]=None, encoding: Optional[str]=None, **args):
        if base is not None:
            warnings.warn('NQuadsSerializer does not support base.')
        if encoding is not None and encoding.lower() != self.encoding.lower():
            warnings.warn(f'NQuadsSerializer does not use custom encoding. Given encoding was: {encoding}')
        encoding = self.encoding
        for context in self.store.contexts():
            for triple in context:
                stream.write(_nq_row(triple, context.identifier).encode(encoding, 'replace'))
        stream.write('\n'.encode('latin-1'))