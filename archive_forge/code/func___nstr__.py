from ..libmp.backend import xrange
import warnings
def __nstr__(self, n=None, **kwargs):
    res = []
    maxlen = [0] * self.cols
    for i in range(self.rows):
        res.append([])
        for j in range(self.cols):
            if n:
                string = self.ctx.nstr(self[i, j], n, **kwargs)
            else:
                string = str(self[i, j])
            res[-1].append(string)
            maxlen[j] = max(len(string), maxlen[j])
    for i, row in enumerate(res):
        for j, elem in enumerate(row):
            row[j] = elem.rjust(maxlen[j])
        res[i] = '[' + colsep.join(row) + ']'
    return rowsep.join(res)