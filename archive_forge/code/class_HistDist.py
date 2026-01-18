from David Huard's scipy sandbox, also attached to a ticket and
import scipy.interpolate as interpolate
import numpy as np
class HistDist:
    """Distribution with piecewise linear cdf, pdf is step function

    can be created from empiricial distribution or from a histogram (not done yet)

    work in progress, not finished


    """

    def __init__(self, data):
        self.data = np.atleast_1d(data)
        self.binlimit = np.array([self.data.min(), self.data.max()])
        sortind = np.argsort(data)
        self._datasorted = data[sortind]
        self.ranking = np.argsort(sortind)
        cdf = self.empiricalcdf()
        self._empcdfsorted = np.sort(cdf)
        self.cdfintp = interpolate.interp1d(self._datasorted, self._empcdfsorted)
        self.ppfintp = interpolate.interp1d(self._empcdfsorted, self._datasorted)

    def empiricalcdf(self, data=None, method='Hazen'):
        """Return the empirical cdf.

        Methods available:
            Hazen:       (i-0.5)/N
                Weibull:     i/(N+1)
            Chegodayev:  (i-.3)/(N+.4)
            Cunnane:     (i-.4)/(N+.2)
            Gringorten:  (i-.44)/(N+.12)
            California:  (i-1)/N

        Where i goes from 1 to N.
        """
        if data is None:
            data = self.data
            i = self.ranking
        else:
            i = np.argsort(np.argsort(data)) + 1.0
        N = len(data)
        method = method.lower()
        if method == 'hazen':
            cdf = (i - 0.5) / N
        elif method == 'weibull':
            cdf = i / (N + 1.0)
        elif method == 'california':
            cdf = (i - 1.0) / N
        elif method == 'chegodayev':
            cdf = (i - 0.3) / (N + 0.4)
        elif method == 'cunnane':
            cdf = (i - 0.4) / (N + 0.2)
        elif method == 'gringorten':
            cdf = (i - 0.44) / (N + 0.12)
        else:
            raise ValueError('Unknown method. Choose among Weibull, Hazen,Chegodayev, Cunnane, Gringorten and California.')
        return cdf

    def cdf_emp(self, score):
        """
        this is score in dh

        """
        return self.cdfintp(score)

    def ppf_emp(self, quantile):
        """
        this is score in dh

        """
        return self.ppfintp(quantile)

    def optimize_binning(self, method='Freedman'):
        """Find the optimal number of bins and update the bin countaccordingly.
        Available methods : Freedman
                            Scott
        """
        nobs = len(self.data)
        if method == 'Freedman':
            IQR = self.ppf_emp(0.75) - self.ppf_emp(0.25)
            width = 2 * IQR * nobs ** (-1.0 / 3)
        elif method == 'Scott':
            width = 3.49 * np.std(self.data) * nobs ** (-1.0 / 3)
        self.nbin = np.ptp(self.binlimit) / width
        return self.nbin