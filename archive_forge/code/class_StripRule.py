import re
import regex
from nltk.chunk.api import ChunkParserI
from nltk.tree import Tree
class StripRule(RegexpChunkRule):
    """
    A rule specifying how to remove strips to a ``ChunkString``,
    using a matching tag pattern.  When applied to a
    ``ChunkString``, it will find any substring that matches this
    tag pattern and that is contained in a chunk, and remove it
    from that chunk, thus creating two new chunks.
    """

    def __init__(self, tag_pattern, descr):
        """
        Construct a new ``StripRule``.

        :type tag_pattern: str
        :param tag_pattern: This rule's tag pattern.  When
            applied to a ``ChunkString``, this rule will
            find any substring that matches this tag pattern and that
            is contained in a chunk, and remove it from that chunk,
            thus creating two new chunks.
        :type descr: str
        :param descr: A short description of the purpose and/or effect
            of this rule.
        """
        self._pattern = tag_pattern
        regexp = re.compile('(?P<strip>%s)%s' % (tag_pattern2re_pattern(tag_pattern), ChunkString.IN_CHUNK_PATTERN))
        RegexpChunkRule.__init__(self, regexp, '}\\g<strip>{', descr)

    def __repr__(self):
        """
        Return a string representation of this rule.  It has the form::

            <StripRule: '<IN|VB.*>'>

        Note that this representation does not include the
        description string; that string can be accessed
        separately with the ``descr()`` method.

        :rtype: str
        """
        return '<StripRule: ' + repr(self._pattern) + '>'