from statsmodels.compat.pandas import FUTURE_STACK
import numpy as np
import pandas as pd
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import fmt_params
@property
def _revision_grouped_impacts(self):
    s = self.revision_grouped_impacts.stack(**FUTURE_STACK)
    df = s.rename('impact').to_frame()
    df = df.reindex(['revision date', 'revised variable', 'impact'], axis=1)
    if self.revisions_details_start > 0:
        df['revision date'] = self.updated.model._index[self.revisions_details_start - 1]
        df['revised variable'] = 'all prior revisions'
    df = df.set_index(['revision date', 'revised variable'], append=True).reorder_levels([2, 3, 0, 1])
    return df