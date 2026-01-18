import re
import regex
from nltk.chunk.api import ChunkParserI
from nltk.tree import Tree
def _notrace_apply(self, chunkstr):
    """
        Apply each rule of this ``RegexpChunkParser`` to ``chunkstr``, in
        turn.

        :param chunkstr: The chunk string to which each rule should be
            applied.
        :type chunkstr: ChunkString
        :rtype: None
        """
    for rule in self._rules:
        rule.apply(chunkstr)