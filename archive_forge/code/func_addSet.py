from io import StringIO
from antlr4.Token import Token
def addSet(self, other: IntervalSet):
    if other.intervals is not None:
        for i in other.intervals:
            self.addRange(i)
    return self