from statsmodels.compat.python import lzip
from io import StringIO
import numpy as np
from statsmodels.iolib import SimpleTable
def _stats_table(self):
    model = self.model
    part2Lstubs = ('No. of Equations:', 'Nobs:', 'Log likelihood:', 'AIC:')
    part2Rstubs = ('BIC:', 'HQIC:', 'FPE:', 'Det(Omega_mle):')
    part2Ldata = [[model.neqs], [model.nobs], [model.llf], [model.aic]]
    part2Rdata = [[model.bic], [model.hqic], [model.fpe], [model.detomega]]
    part2Lheader = None
    part2L = SimpleTable(part2Ldata, part2Lheader, part2Lstubs, txt_fmt=self.part2_fmt)
    part2R = SimpleTable(part2Rdata, part2Lheader, part2Rstubs, txt_fmt=self.part2_fmt)
    part2L.extend_right(part2R)
    return str(part2L)