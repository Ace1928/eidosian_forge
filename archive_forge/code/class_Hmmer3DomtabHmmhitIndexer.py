from itertools import chain
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
from .hmmer3_tab import Hmmer3TabParser, Hmmer3TabIndexer
class Hmmer3DomtabHmmhitIndexer(Hmmer3TabIndexer):
    """HMMER domain table indexer using hit coordinates.

    Indexer class for HMMER domain table output that assumes HMM profile
    coordinates are hit coordinates.
    """
    _parser = Hmmer3DomtabHmmhitParser
    _query_id_idx = 3