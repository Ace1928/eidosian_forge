from pyparsing import (Literal, oneOf, printables, ParserElement, Combine,
class GroupEmitter(object):

    def __init__(self, exprs):
        self.exprs = ParseResults(exprs)

    def makeGenerator(self):

        def groupGen():

            def recurseList(elist):
                if len(elist) == 1:
                    for s in elist[0].makeGenerator()():
                        yield s
                else:
                    for s in elist[0].makeGenerator()():
                        for s2 in recurseList(elist[1:]):
                            yield (s + s2)
            if self.exprs:
                for s in recurseList(self.exprs):
                    yield s
        return groupGen