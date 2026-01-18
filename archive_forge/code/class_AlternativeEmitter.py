from pyparsing import (Literal, oneOf, printables, ParserElement, Combine,
class AlternativeEmitter(object):

    def __init__(self, exprs):
        self.exprs = exprs

    def makeGenerator(self):

        def altGen():
            for e in self.exprs:
                for s in e.makeGenerator()():
                    yield s
        return altGen