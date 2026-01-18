import numpy as np
from scipy import stats
from scipy.special import comb
import warnings
from statsmodels.tools.validation import array_like
class RunsProb:
    """distribution of success runs of length k or more (classical definition)

    The underlying process is assumed to be a sequence of Bernoulli trials
    of a given length n.

    not sure yet, how to interpret or use the distribution for runs
    of length k or more.

    Musseli also has longest success run, and waiting time distribution
    negative binomial of order k and geometric of order k

    need to compare with Godpole

    need a MonteCarlo function to do some quick tests before doing more


    """

    def pdf(self, x, k, n, p):
        """distribution of success runs of length k or more

        Parameters
        ----------
        x : float
            count of runs of length n
        k : int
            length of runs
        n : int
            total number of observations or trials
        p : float
            probability of success in each Bernoulli trial

        Returns
        -------
        pdf : float
            probability that x runs of length of k are observed

        Notes
        -----
        not yet vectorized

        References
        ----------
        Muselli 1996, theorem 3
        """
        q = 1 - p
        m = np.arange(x, (n + 1) // (k + 1) + 1)[:, None]
        terms = (-1) ** (m - x) * comb(m, x) * p ** (m * k) * q ** (m - 1) * (comb(n - m * k, m - 1) + q * comb(n - m * k, m))
        return terms.sum(0)

    def pdf_nb(self, x, k, n, p):
        pass