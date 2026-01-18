import warnings
from typing import IO, Optional
from rdflib.graph import ConjunctiveGraph, Graph
from rdflib.plugins.serializers.nt import _quoteLiteral
from rdflib.serializer import Serializer
from rdflib.term import Literal
def _nq_row(triple, context):
    if isinstance(triple[2], Literal):
        return '%s %s %s %s .\n' % (triple[0].n3(), triple[1].n3(), _quoteLiteral(triple[2]), context.n3())
    else:
        return '%s %s %s %s .\n' % (triple[0].n3(), triple[1].n3(), triple[2].n3(), context.n3())