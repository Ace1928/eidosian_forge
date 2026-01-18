from .graphs import ReducedGraph, Digraph, Poset
from collections import deque
import operator
class WhiteheadMove:
    """
    Holds the data describing a Whitehead move.
    """

    def __init__(self, letter, cut_set, generators, alphabet):
        self.letter = letter
        self.cut_set = cut_set
        self.generators = generators
        self.alphabet = alphabet

    def __repr__(self):
        subs = []
        for x in self.generators:
            sub = '%s -> ' % self.alphabet[x]
            if x == self.letter or x == -self.letter:
                sub += self.alphabet[x]
            else:
                if -x not in self.cut_set and x != self.letter:
                    sub += self.alphabet[self.letter]
                sub += self.alphabet[x]
                if x not in self.cut_set and x != -self.letter:
                    sub += self.alphabet[-self.letter]
            subs.append(sub)
        return ', '.join(subs)