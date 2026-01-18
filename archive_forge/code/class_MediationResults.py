import numpy as np
import pandas as pd
from statsmodels.graphics.utils import maybe_name_or_idx
class MediationResults:
    """
    A class for holding the results of a mediation analysis.

    The following terms are used in the summary output:

    ACME : average causal mediated effect
    ADE : average direct effect
    """

    def __init__(self, indirect_effects, direct_effects):
        self.indirect_effects = indirect_effects
        self.direct_effects = direct_effects
        indirect_effects_avg = [None, None]
        direct_effects_avg = [None, None]
        for t in (0, 1):
            indirect_effects_avg[t] = indirect_effects[t].mean(0)
            direct_effects_avg[t] = direct_effects[t].mean(0)
        self.ACME_ctrl = indirect_effects_avg[0]
        self.ACME_tx = indirect_effects_avg[1]
        self.ADE_ctrl = direct_effects_avg[0]
        self.ADE_tx = direct_effects_avg[1]
        self.total_effect = (self.ACME_ctrl + self.ACME_tx + self.ADE_ctrl + self.ADE_tx) / 2
        self.prop_med_ctrl = self.ACME_ctrl / self.total_effect
        self.prop_med_tx = self.ACME_tx / self.total_effect
        self.prop_med_avg = (self.prop_med_ctrl + self.prop_med_tx) / 2
        self.ACME_avg = (self.ACME_ctrl + self.ACME_tx) / 2
        self.ADE_avg = (self.ADE_ctrl + self.ADE_tx) / 2

    def summary(self, alpha=0.05):
        """
        Provide a summary of a mediation analysis.
        """
        columns = ['Estimate', 'Lower CI bound', 'Upper CI bound', 'P-value']
        index = ['ACME (control)', 'ACME (treated)', 'ADE (control)', 'ADE (treated)', 'Total effect', 'Prop. mediated (control)', 'Prop. mediated (treated)', 'ACME (average)', 'ADE (average)', 'Prop. mediated (average)']
        smry = pd.DataFrame(columns=columns, index=index)
        for i, vec in enumerate([self.ACME_ctrl, self.ACME_tx, self.ADE_ctrl, self.ADE_tx, self.total_effect, self.prop_med_ctrl, self.prop_med_tx, self.ACME_avg, self.ADE_avg, self.prop_med_avg]):
            if vec is self.prop_med_ctrl or vec is self.prop_med_tx or vec is self.prop_med_avg:
                smry.iloc[i, 0] = np.median(vec)
            else:
                smry.iloc[i, 0] = vec.mean()
            smry.iloc[i, 1] = np.percentile(vec, 100 * alpha / 2)
            smry.iloc[i, 2] = np.percentile(vec, 100 * (1 - alpha / 2))
            smry.iloc[i, 3] = _pvalue(vec)
        smry = smry.apply(pd.to_numeric, errors='coerce')
        return smry