from statsmodels.compat.python import lzip
from io import StringIO
import numpy as np
from statsmodels.iolib import SimpleTable
def _coef_table(self):
    model = self.model
    k = model.neqs
    Xnames = self.model.exog_names
    data = lzip(model.params.T.ravel(), model.stderr.T.ravel(), model.tvalues.T.ravel(), model.pvalues.T.ravel())
    header = ('coefficient', 'std. error', 't-stat', 'prob')
    buf = StringIO()
    dim = k * model.k_ar + model.k_trend + model.k_exog_user
    for i in range(k):
        section = 'Results for equation %s' % model.names[i]
        buf.write(section + '\n')
        table = SimpleTable(data[dim * i:dim * (i + 1)], header, Xnames, title=None, txt_fmt=self.default_fmt)
        buf.write(str(table) + '\n')
        if i < k - 1:
            buf.write('\n')
    return buf.getvalue()