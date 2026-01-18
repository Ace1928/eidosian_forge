from statsmodels.compat.python import lrange
import numpy as np
from statsmodels.iolib.table import SimpleTable
class StatTestMC:
    """class to run Monte Carlo study on a statistical test'''

    TODO
    print(summary, for quantiles and for histogram
    draft in trying out script log

    Parameters
    ----------
    dgp : callable
        Function that generates the data to be used in Monte Carlo that should
        return a new sample with each call
    statistic : callable
        Function that calculates the test statistic, which can return either
        a single statistic or a 1d array_like (tuple, list, ndarray).
        see also statindices in description of run

    Attributes
    ----------
    many methods store intermediate results

    self.mcres : ndarray (nrepl, nreturns) or (nrepl, len(statindices))
        Monte Carlo results stored by run


    Notes
    -----

    .. Warning::
       This is (currently) designed for a single call to run. If run is
       called a second time with different arguments, then some attributes might
       not be updated, and, therefore, not correspond to the same run.

    .. Warning::
       Under Construction, do not expect stability in Api or implementation


    Examples
    --------

    Define a function that defines our test statistic:

    def lb(x):
        s,p = acorr_ljungbox(x, lags=4)
        return np.r_[s, p]

    Note lb returns eight values.

    Define a random sample generator, for example 500 independently, normal
    distributed observations in a sample:


    def normalnoisesim(nobs=500, loc=0.0):
        return (loc+np.random.randn(nobs))

    Create instance and run Monte Carlo. Using statindices=list(range(4)) means that
    only the first for values of the return of the statistic (lb) are stored
    in the Monte Carlo results.

    mc1 = StatTestMC(normalnoisesim, lb)
    mc1.run(5000, statindices=list(range(4)))

    Most of the other methods take an idx which indicates for which columns
    the results should be presented, e.g.

    print(mc1.cdf(crit, [1,2,3])[1]
    """

    def __init__(self, dgp, statistic):
        self.dgp = dgp
        self.statistic = statistic

    def run(self, nrepl, statindices=None, dgpargs=[], statsargs=[]):
        """run the actual Monte Carlo and save results

        Parameters
        ----------
        nrepl : int
            number of Monte Carlo repetitions
        statindices : None or list of integers
           determines which values of the return of the statistic
           functions are stored in the Monte Carlo. Default None
           means the entire return. If statindices is a list of
           integers, then it will be used as index into the return.
        dgpargs : tuple
           optional parameters for the DGP
        statsargs : tuple
           optional parameters for the statistics function

        Returns
        -------
        None, all results are attached


        """
        self.nrepl = nrepl
        self.statindices = statindices
        self.dgpargs = dgpargs
        self.statsargs = statsargs
        dgp = self.dgp
        statfun = self.statistic
        mcres0 = statfun(dgp(*dgpargs), *statsargs)
        self.nreturn = nreturns = len(np.ravel(mcres0))
        if statindices is None:
            mcres = np.zeros(nrepl)
            mcres[0] = mcres0
            for ii in range(1, nrepl - 1, nreturns):
                x = dgp(*dgpargs)
                mcres[ii] = statfun(x, *statsargs)
        else:
            self.nreturn = nreturns = len(statindices)
            self.mcres = mcres = np.zeros((nrepl, nreturns))
            mcres[0] = [mcres0[i] for i in statindices]
            for ii in range(1, nrepl - 1):
                x = dgp(*dgpargs)
                ret = statfun(x, *statsargs)
                mcres[ii] = [ret[i] for i in statindices]
        self.mcres = mcres

    def histogram(self, idx=None, critval=None):
        """calculate histogram values

        does not do any plotting

        I do not remember what I wanted here, looks similar to the new cdf
        method, but this also does a binned pdf (self.histo)


        """
        if self.mcres.ndim == 2:
            if idx is not None:
                mcres = self.mcres[:, idx]
            else:
                raise ValueError('currently only 1 statistic at a time')
        else:
            mcres = self.mcres
        if critval is None:
            histo = np.histogram(mcres, bins=10)
        else:
            if not critval[0] == -np.inf:
                bins = np.r_[-np.inf, critval, np.inf]
            if not critval[0] == -np.inf:
                bins = np.r_[bins, np.inf]
            histo = np.histogram(mcres, bins=np.r_[-np.inf, critval, np.inf])
        self.histo = histo
        self.cumhisto = np.cumsum(histo[0]) * 1.0 / self.nrepl
        self.cumhistoreversed = np.cumsum(histo[0][::-1])[::-1] * 1.0 / self.nrepl
        return (histo, self.cumhisto, self.cumhistoreversed)

    def get_mc_sorted(self):
        if not hasattr(self, 'mcressort'):
            self.mcressort = np.sort(self.mcres, axis=0)
        return self.mcressort

    def quantiles(self, idx=None, frac=[0.01, 0.025, 0.05, 0.1, 0.975]):
        """calculate quantiles of Monte Carlo results

        similar to ppf

        Parameters
        ----------
        idx : None or list of integers
            List of indices into the Monte Carlo results (columns) that should
            be used in the calculation
        frac : array_like, float
            Defines which quantiles should be calculated. For example a frac
            of 0.1 finds the 10% quantile, x such that cdf(x)=0.1

        Returns
        -------
        frac : ndarray
            same values as input, TODO: I should drop this again ?
        quantiles : ndarray, (len(frac), len(idx))
            the quantiles with frac in rows and idx variables in columns

        Notes
        -----

        rename to ppf ? make frac required
        change sequence idx, frac


        """
        if self.mcres.ndim == 2:
            if idx is not None:
                mcres = self.mcres[:, idx]
            else:
                raise ValueError('currently only 1 statistic at a time')
        else:
            mcres = self.mcres
        self.frac = frac = np.asarray(frac)
        mc_sorted = self.get_mc_sorted()[:, idx]
        return (frac, mc_sorted[(self.nrepl * frac).astype(int)])

    def cdf(self, x, idx=None):
        """calculate cumulative probabilities of Monte Carlo results

        Parameters
        ----------
        idx : None or list of integers
            List of indices into the Monte Carlo results (columns) that should
            be used in the calculation
        frac : array_like, float
            Defines which quantiles should be calculated. For example a frac
            of 0.1 finds the 10% quantile, x such that cdf(x)=0.1

        Returns
        -------
        x : ndarray
            same as input, TODO: I should drop this again ?
        probs : ndarray, (len(x), len(idx))
            the quantiles with frac in rows and idx variables in columns



        """
        idx = np.atleast_1d(idx).tolist()
        mc_sorted = self.get_mc_sorted()
        x = np.asarray(x)
        if x.ndim > 1 and x.shape[1] == len(idx):
            use_xi = True
        else:
            use_xi = False
        x_ = x
        probs = []
        for i, ix in enumerate(idx):
            if use_xi:
                x_ = x[:, i]
            probs.append(np.searchsorted(mc_sorted[:, ix], x_) / float(self.nrepl))
        probs = np.asarray(probs).T
        return (x, probs)

    def plot_hist(self, idx, distpdf=None, bins=50, ax=None, kwds=None):
        """plot the histogram against a reference distribution

        Parameters
        ----------
        idx : None or list of integers
            List of indices into the Monte Carlo results (columns) that should
            be used in the calculation
        distpdf : callable
            probability density function of reference distribution
        bins : {int, array_like}
            used unchanged for matplotlibs hist call
        ax : TODO: not implemented yet
        kwds : None or tuple of dicts
            extra keyword options to the calls to the matplotlib functions,
            first dictionary is for his, second dictionary for plot of the
            reference distribution

        Returns
        -------
        None


        """
        if kwds is None:
            kwds = ({}, {})
        if self.mcres.ndim == 2:
            if idx is not None:
                mcres = self.mcres[:, idx]
            else:
                raise ValueError('currently only 1 statistic at a time')
        else:
            mcres = self.mcres
        lsp = np.linspace(mcres.min(), mcres.max(), 100)
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.hist(mcres, bins=bins, normed=True, **kwds[0])
        plt.plot(lsp, distpdf(lsp), 'r', **kwds[1])

    def summary_quantiles(self, idx, distppf, frac=[0.01, 0.025, 0.05, 0.1, 0.975], varnames=None, title=None):
        """summary table for quantiles (critical values)

        Parameters
        ----------
        idx : None or list of integers
            List of indices into the Monte Carlo results (columns) that should
            be used in the calculation
        distppf : callable
            probability density function of reference distribution
            TODO: use `crit` values instead or additional, see summary_cdf
        frac : array_like, float
            probabilities for which
        varnames : None, or list of strings
            optional list of variable names, same length as idx

        Returns
        -------
        table : instance of SimpleTable
            use `print(table` to see results

        """
        idx = np.atleast_1d(idx)
        quant, mcq = self.quantiles(idx, frac=frac)
        crit = distppf(np.atleast_2d(quant).T)
        mml = []
        for i, ix in enumerate(idx):
            mml.extend([mcq[:, i], crit[:, i]])
        mmlar = np.column_stack([quant] + mml)
        if title:
            title = title + ' Quantiles (critical values)'
        else:
            title = 'Quantiles (critical values)'
        if varnames is None:
            varnames = ['var%d' % i for i in range(mmlar.shape[1] // 2)]
        headers = ['\nprob'] + ['{}\n{}'.format(i, t) for i in varnames for t in ['mc', 'dist']]
        return SimpleTable(mmlar, txt_fmt={'data_fmts': ['%#6.3f'] + ['%#10.4f'] * (mmlar.shape[1] - 1)}, title=title, headers=headers)

    def summary_cdf(self, idx, frac, crit, varnames=None, title=None):
        """summary table for cumulative density function


        Parameters
        ----------
        idx : None or list of integers
            List of indices into the Monte Carlo results (columns) that should
            be used in the calculation
        frac : array_like, float
            probabilities for which
        crit : array_like
            values for which cdf is calculated
        varnames : None, or list of strings
            optional list of variable names, same length as idx

        Returns
        -------
        table : instance of SimpleTable
            use `print(table` to see results


        """
        idx = np.atleast_1d(idx)
        mml = []
        for i in range(len(idx)):
            mml.append(self.cdf(crit[:, i], [idx[i]])[1].ravel())
        mmlar = np.column_stack([frac] + mml)
        if title:
            title = title + ' Probabilites'
        else:
            title = 'Probabilities'
        if varnames is None:
            varnames = ['var%d' % i for i in range(mmlar.shape[1] - 1)]
        headers = ['prob'] + varnames
        return SimpleTable(mmlar, txt_fmt={'data_fmts': ['%#6.3f'] + ['%#10.4f'] * (np.array(mml).shape[1] - 1)}, title=title, headers=headers)