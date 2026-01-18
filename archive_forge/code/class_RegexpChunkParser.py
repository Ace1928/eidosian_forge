import re
import regex
from nltk.chunk.api import ChunkParserI
from nltk.tree import Tree
class RegexpChunkParser(ChunkParserI):
    """
    A regular expression based chunk parser.  ``RegexpChunkParser`` uses a
    sequence of "rules" to find chunks of a single type within a
    text.  The chunking of the text is encoded using a ``ChunkString``,
    and each rule acts by modifying the chunking in the
    ``ChunkString``.  The rules are all implemented using regular
    expression matching and substitution.

    The ``RegexpChunkRule`` class and its subclasses (``ChunkRule``,
    ``StripRule``, ``UnChunkRule``, ``MergeRule``, and ``SplitRule``)
    define the rules that are used by ``RegexpChunkParser``.  Each rule
    defines an ``apply()`` method, which modifies the chunking encoded
    by a given ``ChunkString``.

    :type _rules: list(RegexpChunkRule)
    :ivar _rules: The list of rules that should be applied to a text.
    :type _trace: int
    :ivar _trace: The default level of tracing.

    """

    def __init__(self, rules, chunk_label='NP', root_label='S', trace=0):
        """
        Construct a new ``RegexpChunkParser``.

        :type rules: list(RegexpChunkRule)
        :param rules: The sequence of rules that should be used to
            generate the chunking for a tagged text.
        :type chunk_label: str
        :param chunk_label: The node value that should be used for
            chunk subtrees.  This is typically a short string
            describing the type of information contained by the chunk,
            such as ``"NP"`` for base noun phrases.
        :type root_label: str
        :param root_label: The node value that should be used for the
            top node of the chunk structure.
        :type trace: int
        :param trace: The level of tracing that should be used when
            parsing a text.  ``0`` will generate no tracing output;
            ``1`` will generate normal tracing output; and ``2`` or
            higher will generate verbose tracing output.
        """
        self._rules = rules
        self._trace = trace
        self._chunk_label = chunk_label
        self._root_label = root_label

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

    def parse(self, chunk_struct, trace=None):
        """
        :type chunk_struct: Tree
        :param chunk_struct: the chunk structure to be (further) chunked
        :type trace: int
        :param trace: The level of tracing that should be used when
            parsing a text.  ``0`` will generate no tracing output;
            ``1`` will generate normal tracing output; and ``2`` or
            higher will generate verbose tracing output.  This value
            overrides the trace level value that was given to the
            constructor.
        :rtype: Tree
        :return: a chunk structure that encodes the chunks in a given
            tagged sentence.  A chunk is a non-overlapping linguistic
            group, such as a noun phrase.  The set of chunks
            identified in the chunk structure depends on the rules
            used to define this ``RegexpChunkParser``.
        """
        if len(chunk_struct) == 0:
            print('Warning: parsing empty text')
            return Tree(self._root_label, [])
        try:
            chunk_struct.label()
        except AttributeError:
            chunk_struct = Tree(self._root_label, chunk_struct)
        if trace is None:
            trace = self._trace
        chunkstr = ChunkString(chunk_struct)
        if trace:
            verbose = trace > 1
            self._trace_apply(chunkstr, verbose)
        else:
            self._notrace_apply(chunkstr)
        return chunkstr.to_chunkstruct(self._chunk_label)

    def rules(self):
        """
        :return: the sequence of rules used by ``RegexpChunkParser``.
        :rtype: list(RegexpChunkRule)
        """
        return self._rules

    def __repr__(self):
        """
        :return: a concise string representation of this
            ``RegexpChunkParser``.
        :rtype: str
        """
        return '<RegexpChunkParser with %d rules>' % len(self._rules)

    def __str__(self):
        """
        :return: a verbose string representation of this ``RegexpChunkParser``.
        :rtype: str
        """
        s = 'RegexpChunkParser with %d rules:\n' % len(self._rules)
        margin = 0
        for rule in self._rules:
            margin = max(margin, len(rule.descr()))
        if margin < 35:
            format = '    %' + repr(-(margin + 3)) + 's%s\n'
        else:
            format = '    %s\n      %s\n'
        for rule in self._rules:
            s += format % (rule.descr(), repr(rule))
        return s[:-1]