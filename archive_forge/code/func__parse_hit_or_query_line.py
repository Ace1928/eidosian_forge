import re
from functools import reduce
from abc import ABC, abstractmethod
from typing import Optional, Type
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
from Bio.SeqUtils import seq1
def _parse_hit_or_query_line(line):
    """Parse the 'Query:' line of exonerate alignment outputs (PRIVATE)."""
    try:
        mark, id, desc = line.split(' ', 2)
    except ValueError:
        mark, id = line.split(' ', 1)
        desc = ''
    return (id, desc)