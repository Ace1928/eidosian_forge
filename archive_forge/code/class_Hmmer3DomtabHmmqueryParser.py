from itertools import chain
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
from .hmmer3_tab import Hmmer3TabParser, Hmmer3TabIndexer
class Hmmer3DomtabHmmqueryParser(Hmmer3DomtabParser):
    """HMMER domain table parser using query coordinates.

    Parser for the HMMER domain table format that assumes HMM profile
    coordinates are query coordinates.
    """
    hmm_as_hit = False