from pystatsmodels mailinglist 20100524
from collections import namedtuple
from statsmodels.compat.python import lzip, lrange
import copy
import math
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from scipy import stats, interpolate
from statsmodels.iolib.table import SimpleTable
from statsmodels.stats.multitest import multipletests, _ecdf as ecdf, fdrcorrection as fdrcorrection0, fdrcorrection_twostage
from statsmodels.graphics import utils
from statsmodels.tools.sm_exceptions import ValueWarning
class MultiComparison:
    """Tests for multiple comparisons

    Parameters
    ----------
    data : ndarray
        independent data samples
    groups : ndarray
        group labels corresponding to each data point
    group_order : list[str], optional
        the desired order for the group mean results to be reported in. If
        not specified, results are reported in increasing order.
        If group_order does not contain all labels that are in groups, then
        only those observations are kept that have a label in group_order.

    """

    def __init__(self, data, groups, group_order=None):
        if len(data) != len(groups):
            raise ValueError('data has %d elements and groups has %d' % (len(data), len(groups)))
        self.data = np.asarray(data)
        self.groups = groups = np.asarray(groups)
        if group_order is None:
            self.groupsunique, self.groupintlab = np.unique(groups, return_inverse=True)
        else:
            for grp in group_order:
                if grp not in groups:
                    raise ValueError("group_order value '%s' not found in groups" % grp)
            self.groupsunique = np.array(group_order)
            self.groupintlab = np.empty(len(data), int)
            self.groupintlab.fill(-999)
            count = 0
            for name in self.groupsunique:
                idx = np.where(self.groups == name)[0]
                count += len(idx)
                self.groupintlab[idx] = np.where(self.groupsunique == name)[0]
            if count != self.data.shape[0]:
                import warnings
                warnings.warn('group_order does not contain all groups:' + ' dropping observations', ValueWarning)
                mask_keep = self.groupintlab != -999
                self.groupintlab = self.groupintlab[mask_keep]
                self.data = self.data[mask_keep]
                self.groups = self.groups[mask_keep]
        if len(self.groupsunique) < 2:
            raise ValueError('2 or more groups required for multiple comparisons')
        self.datali = [self.data[self.groups == k] for k in self.groupsunique]
        self.pairindices = np.triu_indices(len(self.groupsunique), 1)
        self.nobs = self.data.shape[0]
        self.ngroups = len(self.groupsunique)

    def getranks(self):
        """convert data to rankdata and attach


        This creates rankdata as it is used for non-parametric tests, where
        in the case of ties the average rank is assigned.


        """
        self.ranks = GroupsStats(np.column_stack([self.data, self.groupintlab]), useranks=True)
        self.rankdata = self.ranks.groupmeanfilter

    def kruskal(self, pairs=None, multimethod='T'):
        """
        pairwise comparison for kruskal-wallis test

        This is just a reimplementation of scipy.stats.kruskal and does
        not yet use a multiple comparison correction.

        """
        self.getranks()
        tot = self.nobs
        meanranks = self.ranks.groupmean
        groupnobs = self.ranks.groupnobs
        f = tot * (tot + 1.0) / 12.0 / stats.tiecorrect(self.rankdata)
        print('MultiComparison.kruskal')
        for i, j in zip(*self.pairindices):
            pdiff = np.abs(meanranks[i] - meanranks[j])
            se = np.sqrt(f * np.sum(1.0 / groupnobs[[i, j]]))
            Q = pdiff / se
            print(i, j, pdiff, se, pdiff / se, pdiff / se > 2.631)
            print(stats.norm.sf(Q) * 2)
            return stats.norm.sf(Q) * 2

    def allpairtest(self, testfunc, alpha=0.05, method='bonf', pvalidx=1):
        """run a pairwise test on all pairs with multiple test correction

        The statistical test given in testfunc is calculated for all pairs
        and the p-values are adjusted by methods in multipletests. The p-value
        correction is generic and based only on the p-values, and does not
        take any special structure of the hypotheses into account.

        Parameters
        ----------
        testfunc : function
            A test function for two (independent) samples. It is assumed that
            the return value on position pvalidx is the p-value.
        alpha : float
            familywise error rate
        method : str
            This specifies the method for the p-value correction. Any method
            of multipletests is possible.
        pvalidx : int (default: 1)
            position of the p-value in the return of testfunc

        Returns
        -------
        sumtab : SimpleTable instance
            summary table for printing

        errors:  TODO: check if this is still wrong, I think it's fixed.
        results from multipletests are in different order
        pval_corrected can be larger than 1 ???
        """
        res = []
        for i, j in zip(*self.pairindices):
            res.append(testfunc(self.datali[i], self.datali[j]))
        res = np.array(res)
        reject, pvals_corrected, alphacSidak, alphacBonf = multipletests(res[:, pvalidx], alpha=alpha, method=method)
        i1, i2 = self.pairindices
        if pvals_corrected is None:
            resarr = np.array(lzip(self.groupsunique[i1], self.groupsunique[i2], np.round(res[:, 0], 4), np.round(res[:, 1], 4), reject), dtype=[('group1', object), ('group2', object), ('stat', float), ('pval', float), ('reject', np.bool_)])
        else:
            resarr = np.array(lzip(self.groupsunique[i1], self.groupsunique[i2], np.round(res[:, 0], 4), np.round(res[:, 1], 4), np.round(pvals_corrected, 4), reject), dtype=[('group1', object), ('group2', object), ('stat', float), ('pval', float), ('pval_corr', float), ('reject', np.bool_)])
        results_table = SimpleTable(resarr, headers=resarr.dtype.names)
        results_table.title = 'Test Multiple Comparison %s \n%s%4.2f method=%s' % (testfunc.__name__, 'FWER=', alpha, method) + '\nalphacSidak=%4.2f, alphacBonf=%5.3f' % (alphacSidak, alphacBonf)
        return (results_table, (res, reject, pvals_corrected, alphacSidak, alphacBonf), resarr)

    def tukeyhsd(self, alpha=0.05):
        """
        Tukey's range test to compare means of all pairs of groups

        Parameters
        ----------
        alpha : float, optional
            Value of FWER at which to calculate HSD.

        Returns
        -------
        results : TukeyHSDResults instance
            A results class containing relevant data and some post-hoc
            calculations
        """
        self.groupstats = GroupsStats(np.column_stack([self.data, self.groupintlab]), useranks=False)
        gmeans = self.groupstats.groupmean
        gnobs = self.groupstats.groupnobs
        var_ = np.var(self.groupstats.groupdemean(), ddof=len(gmeans))
        res = tukeyhsd(gmeans, gnobs, var_, df=None, alpha=alpha, q_crit=None)
        resarr = np.array(lzip(self.groupsunique[res[0][0]], self.groupsunique[res[0][1]], np.round(res[2], 4), np.round(res[8], 4), np.round(res[4][:, 0], 4), np.round(res[4][:, 1], 4), res[1]), dtype=[('group1', object), ('group2', object), ('meandiff', float), ('p-adj', float), ('lower', float), ('upper', float), ('reject', np.bool_)])
        results_table = SimpleTable(resarr, headers=resarr.dtype.names)
        results_table.title = 'Multiple Comparison of Means - Tukey HSD, ' + 'FWER=%4.2f' % alpha
        return TukeyHSDResults(self, results_table, res[5], res[1], res[2], res[3], res[4], res[6], res[7], var_, res[8])