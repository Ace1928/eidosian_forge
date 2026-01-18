from io import StringIO
from antlr4.Token import Token
def addRange(self, v: range):
    if self.intervals is None:
        self.intervals = list()
        self.intervals.append(v)
    else:
        k = 0
        for i in self.intervals:
            if v.stop < i.start:
                self.intervals.insert(k, v)
                return
            elif v.stop == i.start:
                self.intervals[k] = range(v.start, i.stop)
                return
            elif v.start <= i.stop:
                self.intervals[k] = range(min(i.start, v.start), max(i.stop, v.stop))
                self.reduce(k)
                return
            k += 1
        self.intervals.append(v)