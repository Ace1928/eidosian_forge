from typing import (
from blib2to3.pgen2.grammar import Grammar
import sys
from io import StringIO
class NegatedPattern(BasePattern):

    def __init__(self, content: Optional[BasePattern]=None) -> None:
        """
        Initializer.

        The argument is either a pattern or None.  If it is None, this
        only matches an empty sequence (effectively '$' in regex
        lingo).  If it is not None, this matches whenever the argument
        pattern doesn't have any matches.
        """
        if content is not None:
            assert isinstance(content, BasePattern), repr(content)
        self.content = content

    def match(self, node, results=None) -> bool:
        return False

    def match_seq(self, nodes, results=None) -> bool:
        return len(nodes) == 0

    def generate_matches(self, nodes: List[NL]) -> Iterator[Tuple[int, _Results]]:
        if self.content is None:
            if len(nodes) == 0:
                yield (0, {})
        else:
            for c, r in self.content.generate_matches(nodes):
                return
            yield (0, {})