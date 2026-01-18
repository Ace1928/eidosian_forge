import numpy as np
from scipy.stats import f as fdist
from scipy.stats import t as student_t
from scipy import stats
from statsmodels.tools.tools import clean0, fullrank
from statsmodels.stats.multitest import multipletests
class WaldTestResults:

    def __init__(self, statistic, distribution, dist_args, table=None, pvalues=None):
        self.table = table
        self.distribution = distribution
        self.statistic = statistic
        self.dist_args = dist_args
        if table is not None:
            self.statistic = table['statistic'].values
            self.pvalues = table['pvalue'].values
            self.df_constraints = table['df_constraint'].values
            if self.distribution == 'F':
                self.df_denom = table['df_denom'].values
        else:
            if self.distribution == 'chi2':
                self.dist = stats.chi2
                self.df_constraints = self.dist_args[0]
            elif self.distribution == 'F':
                self.dist = stats.f
                self.df_constraints, self.df_denom = self.dist_args
            else:
                raise ValueError('only F and chi2 are possible distribution')
            if pvalues is None:
                self.pvalues = self.dist.sf(np.abs(statistic), *dist_args)
            else:
                self.pvalues = pvalues

    @property
    def col_names(self):
        """column names for summary table
        """
        pr_test = 'P>%s' % self.distribution
        col_names = [self.distribution, pr_test, 'df constraint']
        if self.distribution == 'F':
            col_names.append('df denom')
        return col_names

    def summary_frame(self):
        if hasattr(self, '_dframe'):
            return self._dframe
        renaming = dict(zip(self.table.columns, self.col_names))
        self.dframe = self.table.rename(columns=renaming)
        return self.dframe

    def __str__(self):
        return self.summary_frame().to_string()

    def __repr__(self):
        return str(self.__class__) + '\n' + self.__str__()