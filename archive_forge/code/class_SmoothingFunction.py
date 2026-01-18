import math
import sys
import warnings
from collections import Counter
from fractions import Fraction
from nltk.util import ngrams
class SmoothingFunction:
    """
    This is an implementation of the smoothing techniques
    for segment-level BLEU scores that was presented in
    Boxing Chen and Collin Cherry (2014) A Systematic Comparison of
    Smoothing Techniques for Sentence-Level BLEU. In WMT14.
    http://acl2014.org/acl2014/W14-33/pdf/W14-3346.pdf
    """

    def __init__(self, epsilon=0.1, alpha=5, k=5):
        """
        This will initialize the parameters required for the various smoothing
        techniques, the default values are set to the numbers used in the
        experiments from Chen and Cherry (2014).

        >>> hypothesis1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which', 'ensures',
        ...                 'that', 'the', 'military', 'always', 'obeys', 'the',
        ...                 'commands', 'of', 'the', 'party']
        >>> reference1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that', 'ensures',
        ...               'that', 'the', 'military', 'will', 'forever', 'heed',
        ...               'Party', 'commands']

        >>> chencherry = SmoothingFunction()
        >>> print(sentence_bleu([reference1], hypothesis1)) # doctest: +ELLIPSIS
        0.4118...
        >>> print(sentence_bleu([reference1], hypothesis1, smoothing_function=chencherry.method0)) # doctest: +ELLIPSIS
        0.4118...
        >>> print(sentence_bleu([reference1], hypothesis1, smoothing_function=chencherry.method1)) # doctest: +ELLIPSIS
        0.4118...
        >>> print(sentence_bleu([reference1], hypothesis1, smoothing_function=chencherry.method2)) # doctest: +ELLIPSIS
        0.4452...
        >>> print(sentence_bleu([reference1], hypothesis1, smoothing_function=chencherry.method3)) # doctest: +ELLIPSIS
        0.4118...
        >>> print(sentence_bleu([reference1], hypothesis1, smoothing_function=chencherry.method4)) # doctest: +ELLIPSIS
        0.4118...
        >>> print(sentence_bleu([reference1], hypothesis1, smoothing_function=chencherry.method5)) # doctest: +ELLIPSIS
        0.4905...
        >>> print(sentence_bleu([reference1], hypothesis1, smoothing_function=chencherry.method6)) # doctest: +ELLIPSIS
        0.4135...
        >>> print(sentence_bleu([reference1], hypothesis1, smoothing_function=chencherry.method7)) # doctest: +ELLIPSIS
        0.4905...

        :param epsilon: the epsilon value use in method 1
        :type epsilon: float
        :param alpha: the alpha value use in method 6
        :type alpha: int
        :param k: the k value use in method 4
        :type k: int
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.k = k

    def method0(self, p_n, *args, **kwargs):
        """
        No smoothing.
        """
        p_n_new = []
        for i, p_i in enumerate(p_n):
            if p_i.numerator != 0:
                p_n_new.append(p_i)
            else:
                _msg = str('\nThe hypothesis contains 0 counts of {}-gram overlaps.\nTherefore the BLEU score evaluates to 0, independently of\nhow many N-gram overlaps of lower order it contains.\nConsider using lower n-gram order or use SmoothingFunction()').format(i + 1)
                warnings.warn(_msg)
                p_n_new.append(sys.float_info.min)
        return p_n_new

    def method1(self, p_n, *args, **kwargs):
        """
        Smoothing method 1: Add *epsilon* counts to precision with 0 counts.
        """
        return [(p_i.numerator + self.epsilon) / p_i.denominator if p_i.numerator == 0 else p_i for p_i in p_n]

    def method2(self, p_n, *args, **kwargs):
        """
        Smoothing method 2: Add 1 to both numerator and denominator from
        Chin-Yew Lin and Franz Josef Och (2004) ORANGE: a Method for
        Evaluating Automatic Evaluation Metrics for Machine Translation.
        In COLING 2004.
        """
        return [Fraction(p_n[i].numerator + 1, p_n[i].denominator + 1, _normalize=False) if i != 0 else p_n[0] for i in range(len(p_n))]

    def method3(self, p_n, *args, **kwargs):
        """
        Smoothing method 3: NIST geometric sequence smoothing
        The smoothing is computed by taking 1 / ( 2^k ), instead of 0, for each
        precision score whose matching n-gram count is null.
        k is 1 for the first 'n' value for which the n-gram match count is null/

        For example, if the text contains:

        - one 2-gram match
        - and (consequently) two 1-gram matches

        the n-gram count for each individual precision score would be:

        - n=1  =>  prec_count = 2     (two unigrams)
        - n=2  =>  prec_count = 1     (one bigram)
        - n=3  =>  prec_count = 1/2   (no trigram,  taking 'smoothed' value of 1 / ( 2^k ), with k=1)
        - n=4  =>  prec_count = 1/4   (no fourgram, taking 'smoothed' value of 1 / ( 2^k ), with k=2)
        """
        incvnt = 1
        for i, p_i in enumerate(p_n):
            if p_i.numerator == 0:
                p_n[i] = 1 / (2 ** incvnt * p_i.denominator)
                incvnt += 1
        return p_n

    def method4(self, p_n, references, hypothesis, hyp_len=None, *args, **kwargs):
        """
        Smoothing method 4:
        Shorter translations may have inflated precision values due to having
        smaller denominators; therefore, we give them proportionally
        smaller smoothed counts. Instead of scaling to 1/(2^k), Chen and Cherry
        suggests dividing by 1/ln(len(T)), where T is the length of the translation.
        """
        incvnt = 1
        hyp_len = hyp_len if hyp_len else len(hypothesis)
        for i, p_i in enumerate(p_n):
            if p_i.numerator == 0 and hyp_len > 1:
                numerator = 1 / (2 ** incvnt * self.k / math.log(hyp_len))
                p_n[i] = numerator / p_i.denominator
                incvnt += 1
        return p_n

    def method5(self, p_n, references, hypothesis, hyp_len=None, *args, **kwargs):
        """
        Smoothing method 5:
        The matched counts for similar values of n should be similar. To a
        calculate the n-gram matched count, it averages the n−1, n and n+1 gram
        matched counts.
        """
        hyp_len = hyp_len if hyp_len else len(hypothesis)
        m = {}
        p_n_plus1 = p_n + [modified_precision(references, hypothesis, 5)]
        m[-1] = p_n[0] + 1
        for i, p_i in enumerate(p_n):
            p_n[i] = (m[i - 1] + p_i + p_n_plus1[i + 1]) / 3
            m[i] = p_n[i]
        return p_n

    def method6(self, p_n, references, hypothesis, hyp_len=None, *args, **kwargs):
        """
        Smoothing method 6:
        Interpolates the maximum likelihood estimate of the precision *p_n* with
        a prior estimate *pi0*. The prior is estimated by assuming that the ratio
        between pn and pn−1 will be the same as that between pn−1 and pn−2; from
        Gao and He (2013) Training MRF-Based Phrase Translation Models using
        Gradient Ascent. In NAACL.
        """
        hyp_len = hyp_len if hyp_len else len(hypothesis)
        assert p_n[2], 'This smoothing method requires non-zero precision for bigrams.'
        for i, p_i in enumerate(p_n):
            if i in [0, 1]:
                continue
            else:
                pi0 = 0 if p_n[i - 2] == 0 else p_n[i - 1] ** 2 / p_n[i - 2]
                m = p_i.numerator
                l = sum((1 for _ in ngrams(hypothesis, i + 1)))
                p_n[i] = (m + self.alpha * pi0) / (l + self.alpha)
        return p_n

    def method7(self, p_n, references, hypothesis, hyp_len=None, *args, **kwargs):
        """
        Smoothing method 7:
        Interpolates methods 4 and 5.
        """
        hyp_len = hyp_len if hyp_len else len(hypothesis)
        p_n = self.method4(p_n, references, hypothesis, hyp_len)
        p_n = self.method5(p_n, references, hypothesis, hyp_len)
        return p_n