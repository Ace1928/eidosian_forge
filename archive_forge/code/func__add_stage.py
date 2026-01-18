import re
import regex
from nltk.chunk.api import ChunkParserI
from nltk.tree import Tree
def _add_stage(self, rules, lhs, root_label, trace):
    """
        Helper function for __init__: add a new stage to the parser.
        """
    if rules != []:
        if not lhs:
            raise ValueError('Expected stage marker (eg NP:)')
        parser = RegexpChunkParser(rules, chunk_label=lhs, root_label=root_label, trace=trace)
        self._stages.append(parser)