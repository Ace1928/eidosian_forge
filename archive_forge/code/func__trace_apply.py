import re
import regex
from nltk.chunk.api import ChunkParserI
from nltk.tree import Tree
def _trace_apply(self, chunkstr, verbose):
    """
        Apply each rule of this ``RegexpChunkParser`` to ``chunkstr``, in
        turn.  Generate trace output between each rule.  If ``verbose``
        is true, then generate verbose output.

        :type chunkstr: ChunkString
        :param chunkstr: The chunk string to which each rule should be
            applied.
        :type verbose: bool
        :param verbose: Whether output should be verbose.
        :rtype: None
        """
    print('# Input:')
    print(chunkstr)
    for rule in self._rules:
        rule.apply(chunkstr)
        if verbose:
            print('#', rule.descr() + ' (' + repr(rule) + '):')
        else:
            print('#', rule.descr() + ':')
        print(chunkstr)