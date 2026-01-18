import math as _math
from abc import ABCMeta, abstractmethod
from functools import reduce
class NgramAssocMeasures(metaclass=ABCMeta):
    """
    An abstract class defining a collection of generic association measures.
    Each public method returns a score, taking the following arguments::

        score_fn(count_of_ngram,
                 (count_of_n-1gram_1, ..., count_of_n-1gram_j),
                 (count_of_n-2gram_1, ..., count_of_n-2gram_k),
                 ...,
                 (count_of_1gram_1, ..., count_of_1gram_n),
                 count_of_total_words)

    See ``BigramAssocMeasures`` and ``TrigramAssocMeasures``

    Inheriting classes should define a property _n, and a method _contingency
    which calculates contingency values from marginals in order for all
    association measures defined here to be usable.
    """
    _n = 0

    @staticmethod
    @abstractmethod
    def _contingency(*marginals):
        """Calculates values of a contingency table from marginal values."""
        raise NotImplementedError('The contingency table is not availablein the general ngram case')

    @staticmethod
    @abstractmethod
    def _marginals(*contingency):
        """Calculates values of contingency table marginals from its values."""
        raise NotImplementedError('The contingency table is not availablein the general ngram case')

    @classmethod
    def _expected_values(cls, cont):
        """Calculates expected values for a contingency table."""
        n_all = sum(cont)
        bits = [1 << i for i in range(cls._n)]
        for i in range(len(cont)):
            yield (_product((sum((cont[x] for x in range(2 ** cls._n) if x & j == i & j)) for j in bits)) / n_all ** (cls._n - 1))

    @staticmethod
    def raw_freq(*marginals):
        """Scores ngrams by their frequency"""
        return marginals[NGRAM] / marginals[TOTAL]

    @classmethod
    def student_t(cls, *marginals):
        """Scores ngrams using Student's t test with independence hypothesis
        for unigrams, as in Manning and Schutze 5.3.1.
        """
        return (marginals[NGRAM] - _product(marginals[UNIGRAMS]) / marginals[TOTAL] ** (cls._n - 1)) / (marginals[NGRAM] + _SMALL) ** 0.5

    @classmethod
    def chi_sq(cls, *marginals):
        """Scores ngrams using Pearson's chi-square as in Manning and Schutze
        5.3.3.
        """
        cont = cls._contingency(*marginals)
        exps = cls._expected_values(cont)
        return sum(((obs - exp) ** 2 / (exp + _SMALL) for obs, exp in zip(cont, exps)))

    @staticmethod
    def mi_like(*marginals, **kwargs):
        """Scores ngrams using a variant of mutual information. The keyword
        argument power sets an exponent (default 3) for the numerator. No
        logarithm of the result is calculated.
        """
        return marginals[NGRAM] ** kwargs.get('power', 3) / _product(marginals[UNIGRAMS])

    @classmethod
    def pmi(cls, *marginals):
        """Scores ngrams by pointwise mutual information, as in Manning and
        Schutze 5.4.
        """
        return _log2(marginals[NGRAM] * marginals[TOTAL] ** (cls._n - 1)) - _log2(_product(marginals[UNIGRAMS]))

    @classmethod
    def likelihood_ratio(cls, *marginals):
        """Scores ngrams using likelihood ratios as in Manning and Schutze 5.3.4."""
        cont = cls._contingency(*marginals)
        return 2 * sum((obs * _ln(obs / (exp + _SMALL) + _SMALL) for obs, exp in zip(cont, cls._expected_values(cont))))

    @classmethod
    def poisson_stirling(cls, *marginals):
        """Scores ngrams using the Poisson-Stirling measure."""
        exp = _product(marginals[UNIGRAMS]) / marginals[TOTAL] ** (cls._n - 1)
        return marginals[NGRAM] * (_log2(marginals[NGRAM] / exp) - 1)

    @classmethod
    def jaccard(cls, *marginals):
        """Scores ngrams using the Jaccard index."""
        cont = cls._contingency(*marginals)
        return cont[0] / sum(cont[:-1])