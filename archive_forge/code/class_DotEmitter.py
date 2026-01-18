from pyparsing import (Literal, oneOf, printables, ParserElement, Combine,
class DotEmitter(object):

    def makeGenerator(self):

        def dotGen():
            for c in printables:
                yield c
        return dotGen