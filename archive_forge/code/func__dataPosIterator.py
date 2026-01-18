import re
import itertools
def _dataPosIterator(self):
    cols = itertools.chain(range(self.moduleCount - 1, 6, -2), range(5, 0, -2))
    rows = (list(range(9, self.moduleCount - 8)), list(itertools.chain(range(6), range(7, self.moduleCount))), list(range(9, self.moduleCount)))
    rrows = tuple((list(reversed(r)) for r in rows))
    ppos = QRUtil.getPatternPosition(self.version)
    ppos = set(itertools.chain.from_iterable(((p - 2, p - 1, p, p + 1, p + 2) for p in ppos)))
    maxpos = self.moduleCount - 11
    for col in cols:
        rows, rrows = (rrows, rows)
        if col <= 8:
            rowidx = 0
        elif col >= self.moduleCount - 8:
            rowidx = 2
        else:
            rowidx = 1
        for row in rows[rowidx]:
            for c in range(2):
                c = col - c
                if self.version >= 7:
                    if row < 6 and c >= self.moduleCount - 11:
                        continue
                    elif col < 6 and row >= self.moduleCount - 11:
                        continue
                if row in ppos and c in ppos:
                    if not (row < 11 and (c < 11 or c > maxpos) or (c < 11 and (row < 11 or row > maxpos))):
                        continue
                yield (c, row)