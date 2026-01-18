from itertools import chain
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
from .hmmer3_tab import Hmmer3TabParser, Hmmer3TabIndexer
class Hmmer3DomtabHmmhitParser(Hmmer3DomtabParser):
    """HMMER domain table parser using hit coordinates.

    Parser for the HMMER domain table format that assumes HMM profile
    coordinates are hit coordinates.
    """
    hmm_as_hit = True