import math as _math
from abc import ABCMeta, abstractmethod
from functools import reduce
class BigramAssocMeasures(NgramAssocMeasures):
    """
    A collection of bigram association measures. Each association measure
    is provided as a function with three arguments::

        bigram_score_fn(n_ii, (n_ix, n_xi), n_xx)

    The arguments constitute the marginals of a contingency table, counting
    the occurrences of particular events in a corpus. The letter i in the
    suffix refers to the appearance of the word in question, while x indicates
    the appearance of any word. Thus, for example:

    - n_ii counts ``(w1, w2)``, i.e. the bigram being scored
    - n_ix counts ``(w1, *)``
    - n_xi counts ``(*, w2)``
    - n_xx counts ``(*, *)``, i.e. any bigram

    This may be shown with respect to a contingency table::

                w1    ~w1
             ------ ------
         w2 | n_ii | n_oi | = n_xi
             ------ ------
        ~w2 | n_io | n_oo |
             ------ ------
             = n_ix        TOTAL = n_xx
    """
    _n = 2

    @staticmethod
    def _contingency(n_ii, n_ix_xi_tuple, n_xx):
        """Calculates values of a bigram contingency table from marginal values."""
        n_ix, n_xi = n_ix_xi_tuple
        n_oi = n_xi - n_ii
        n_io = n_ix - n_ii
        return (n_ii, n_oi, n_io, n_xx - n_ii - n_oi - n_io)

    @staticmethod
    def _marginals(n_ii, n_oi, n_io, n_oo):
        """Calculates values of contingency table marginals from its values."""
        return (n_ii, (n_oi + n_ii, n_io + n_ii), n_oo + n_oi + n_io + n_ii)

    @staticmethod
    def _expected_values(cont):
        """Calculates expected values for a contingency table."""
        n_xx = sum(cont)
        for i in range(4):
            yield ((cont[i] + cont[i ^ 1]) * (cont[i] + cont[i ^ 2]) / n_xx)

    @classmethod
    def phi_sq(cls, *marginals):
        """Scores bigrams using phi-square, the square of the Pearson correlation
        coefficient.
        """
        n_ii, n_io, n_oi, n_oo = cls._contingency(*marginals)
        return (n_ii * n_oo - n_io * n_oi) ** 2 / ((n_ii + n_io) * (n_ii + n_oi) * (n_io + n_oo) * (n_oi + n_oo))

    @classmethod
    def chi_sq(cls, n_ii, n_ix_xi_tuple, n_xx):
        """Scores bigrams using chi-square, i.e. phi-sq multiplied by the number
        of bigrams, as in Manning and Schutze 5.3.3.
        """
        n_ix, n_xi = n_ix_xi_tuple
        return n_xx * cls.phi_sq(n_ii, (n_ix, n_xi), n_xx)

    @classmethod
    def fisher(cls, *marginals):
        """Scores bigrams using Fisher's Exact Test (Pedersen 1996).  Less
        sensitive to small counts than PMI or Chi Sq, but also more expensive
        to compute. Requires scipy.
        """
        n_ii, n_io, n_oi, n_oo = cls._contingency(*marginals)
        odds, pvalue = fisher_exact([[n_ii, n_io], [n_oi, n_oo]], alternative='less')
        return pvalue

    @staticmethod
    def dice(n_ii, n_ix_xi_tuple, n_xx):
        """Scores bigrams using Dice's coefficient."""
        n_ix, n_xi = n_ix_xi_tuple
        return 2 * n_ii / (n_ix + n_xi)