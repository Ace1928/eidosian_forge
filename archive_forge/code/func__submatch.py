from typing import (
from blib2to3.pgen2.grammar import Grammar
import sys
from io import StringIO
def _submatch(self, node, results=None) -> bool:
    """
        Match the pattern's content to the node's children.

        This assumes the node type matches and self.content is not None.

        Returns True if it matches, False if not.

        If results is not None, it must be a dict which will be
        updated with the nodes matching named subpatterns.

        When returning False, the results dict may still be updated.
        """
    if self.wildcards:
        for c, r in generate_matches(self.content, node.children):
            if c == len(node.children):
                if results is not None:
                    results.update(r)
                return True
        return False
    if len(self.content) != len(node.children):
        return False
    for subpattern, child in zip(self.content, node.children):
        if not subpattern.match(child, results):
            return False
    return True