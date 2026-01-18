from pyparsing import (Literal, oneOf, printables, ParserElement, Combine,
class CharacterRangeEmitter(object):

    def __init__(self, chars):
        seen = set()
        self.charset = ''.join((seen.add(c) or c for c in chars if c not in seen))

    def __str__(self):
        return '[' + self.charset + ']'

    def __repr__(self):
        return '[' + self.charset + ']'

    def makeGenerator(self):

        def genChars():
            for s in self.charset:
                yield s
        return genChars