import array
import math
import random
import warnings
from abc import ABCMeta, abstractmethod
from collections import Counter, defaultdict
from functools import reduce
from nltk.internals import raise_unorderable_types
class FreqDist(Counter):
    """
    A frequency distribution for the outcomes of an experiment.  A
    frequency distribution records the number of times each outcome of
    an experiment has occurred.  For example, a frequency distribution
    could be used to record the frequency of each word type in a
    document.  Formally, a frequency distribution can be defined as a
    function mapping from each sample to the number of times that
    sample occurred as an outcome.

    Frequency distributions are generally constructed by running a
    number of experiments, and incrementing the count for a sample
    every time it is an outcome of an experiment.  For example, the
    following code will produce a frequency distribution that encodes
    how often each word occurs in a text:

        >>> from nltk.tokenize import word_tokenize
        >>> from nltk.probability import FreqDist
        >>> sent = 'This is an example sentence'
        >>> fdist = FreqDist()
        >>> for word in word_tokenize(sent):
        ...    fdist[word.lower()] += 1

    An equivalent way to do this is with the initializer:

        >>> fdist = FreqDist(word.lower() for word in word_tokenize(sent))

    """

    def __init__(self, samples=None):
        """
        Construct a new frequency distribution.  If ``samples`` is
        given, then the frequency distribution will be initialized
        with the count of each object in ``samples``; otherwise, it
        will be initialized to be empty.

        In particular, ``FreqDist()`` returns an empty frequency
        distribution; and ``FreqDist(samples)`` first creates an empty
        frequency distribution, and then calls ``update`` with the
        list ``samples``.

        :param samples: The samples to initialize the frequency
            distribution with.
        :type samples: Sequence
        """
        Counter.__init__(self, samples)
        self._N = None

    def N(self):
        """
        Return the total number of sample outcomes that have been
        recorded by this FreqDist.  For the number of unique
        sample values (or bins) with counts greater than zero, use
        ``FreqDist.B()``.

        :rtype: int
        """
        if self._N is None:
            self._N = sum(self.values())
        return self._N

    def __setitem__(self, key, val):
        """
        Override ``Counter.__setitem__()`` to invalidate the cached N
        """
        self._N = None
        super().__setitem__(key, val)

    def __delitem__(self, key):
        """
        Override ``Counter.__delitem__()`` to invalidate the cached N
        """
        self._N = None
        super().__delitem__(key)

    def update(self, *args, **kwargs):
        """
        Override ``Counter.update()`` to invalidate the cached N
        """
        self._N = None
        super().update(*args, **kwargs)

    def setdefault(self, key, val):
        """
        Override ``Counter.setdefault()`` to invalidate the cached N
        """
        self._N = None
        super().setdefault(key, val)

    def B(self):
        """
        Return the total number of sample values (or "bins") that
        have counts greater than zero.  For the total
        number of sample outcomes recorded, use ``FreqDist.N()``.
        (FreqDist.B() is the same as len(FreqDist).)

        :rtype: int
        """
        return len(self)

    def hapaxes(self):
        """
        Return a list of all samples that occur once (hapax legomena)

        :rtype: list
        """
        return [item for item in self if self[item] == 1]

    def Nr(self, r, bins=None):
        return self.r_Nr(bins)[r]

    def r_Nr(self, bins=None):
        """
        Return the dictionary mapping r to Nr, the number of samples with frequency r, where Nr > 0.

        :type bins: int
        :param bins: The number of possible sample outcomes.  ``bins``
            is used to calculate Nr(0).  In particular, Nr(0) is
            ``bins-self.B()``.  If ``bins`` is not specified, it
            defaults to ``self.B()`` (so Nr(0) will be 0).
        :rtype: int
        """
        _r_Nr = defaultdict(int)
        for count in self.values():
            _r_Nr[count] += 1
        _r_Nr[0] = bins - self.B() if bins is not None else 0
        return _r_Nr

    def _cumulative_frequencies(self, samples):
        """
        Return the cumulative frequencies of the specified samples.
        If no samples are specified, all counts are returned, starting
        with the largest.

        :param samples: the samples whose frequencies should be returned.
        :type samples: any
        :rtype: list(float)
        """
        cf = 0.0
        for sample in samples:
            cf += self[sample]
            yield cf

    def freq(self, sample):
        """
        Return the frequency of a given sample.  The frequency of a
        sample is defined as the count of that sample divided by the
        total number of sample outcomes that have been recorded by
        this FreqDist.  The count of a sample is defined as the
        number of times that sample outcome was recorded by this
        FreqDist.  Frequencies are always real numbers in the range
        [0, 1].

        :param sample: the sample whose frequency
               should be returned.
        :type sample: any
        :rtype: float
        """
        n = self.N()
        if n == 0:
            return 0
        return self[sample] / n

    def max(self):
        """
        Return the sample with the greatest number of outcomes in this
        frequency distribution.  If two or more samples have the same
        number of outcomes, return one of them; which sample is
        returned is undefined.  If no outcomes have occurred in this
        frequency distribution, return None.

        :return: The sample with the maximum number of outcomes in this
                frequency distribution.
        :rtype: any or None
        """
        if len(self) == 0:
            raise ValueError('A FreqDist must have at least one sample before max is defined.')
        return self.most_common(1)[0][0]

    def plot(self, *args, title='', cumulative=False, percents=False, show=True, **kwargs):
        """
        Plot samples from the frequency distribution
        displaying the most frequent sample first.  If an integer
        parameter is supplied, stop after this many samples have been
        plotted.  For a cumulative plot, specify cumulative=True. Additional
        ``**kwargs`` are passed to matplotlib's plot function.
        (Requires Matplotlib to be installed.)

        :param title: The title for the graph.
        :type title: str
        :param cumulative: Whether the plot is cumulative. (default = False)
        :type cumulative: bool
        :param percents: Whether the plot uses percents instead of counts. (default = False)
        :type percents: bool
        :param show: Whether to show the plot, or only return the ax.
        :type show: bool
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ValueError('The plot function requires matplotlib to be installed.See https://matplotlib.org/') from e
        if len(args) == 0:
            args = [len(self)]
        samples = [item for item, _ in self.most_common(*args)]
        if cumulative:
            freqs = list(self._cumulative_frequencies(samples))
            ylabel = 'Cumulative '
        else:
            freqs = [self[sample] for sample in samples]
            ylabel = ''
        if percents:
            freqs = [f / self.N() * 100 for f in freqs]
            ylabel += 'Percents'
        else:
            ylabel += 'Counts'
        ax = plt.gca()
        ax.grid(True, color='silver')
        if 'linewidth' not in kwargs:
            kwargs['linewidth'] = 2
        if title:
            ax.set_title(title)
        ax.plot(freqs, **kwargs)
        ax.set_xticks(range(len(samples)))
        ax.set_xticklabels([str(s) for s in samples], rotation=90)
        ax.set_xlabel('Samples')
        ax.set_ylabel(ylabel)
        if show:
            plt.show()
        return ax

    def tabulate(self, *args, **kwargs):
        """
        Tabulate the given samples from the frequency distribution (cumulative),
        displaying the most frequent sample first.  If an integer
        parameter is supplied, stop after this many samples have been
        plotted.

        :param samples: The samples to plot (default is all samples)
        :type samples: list
        :param cumulative: A flag to specify whether the freqs are cumulative (default = False)
        :type title: bool
        """
        if len(args) == 0:
            args = [len(self)]
        samples = _get_kwarg(kwargs, 'samples', [item for item, _ in self.most_common(*args)])
        cumulative = _get_kwarg(kwargs, 'cumulative', False)
        if cumulative:
            freqs = list(self._cumulative_frequencies(samples))
        else:
            freqs = [self[sample] for sample in samples]
        width = max((len(f'{s}') for s in samples))
        width = max(width, max((len('%d' % f) for f in freqs)))
        for i in range(len(samples)):
            print('%*s' % (width, samples[i]), end=' ')
        print()
        for i in range(len(samples)):
            print('%*d' % (width, freqs[i]), end=' ')
        print()

    def copy(self):
        """
        Create a copy of this frequency distribution.

        :rtype: FreqDist
        """
        return self.__class__(self)

    def __add__(self, other):
        """
        Add counts from two counters.

        >>> FreqDist('abbb') + FreqDist('bcc')
        FreqDist({'b': 4, 'c': 2, 'a': 1})

        """
        return self.__class__(super().__add__(other))

    def __sub__(self, other):
        """
        Subtract count, but keep only results with positive counts.

        >>> FreqDist('abbbc') - FreqDist('bccd')
        FreqDist({'b': 2, 'a': 1})

        """
        return self.__class__(super().__sub__(other))

    def __or__(self, other):
        """
        Union is the maximum of value in either of the input counters.

        >>> FreqDist('abbb') | FreqDist('bcc')
        FreqDist({'b': 3, 'c': 2, 'a': 1})

        """
        return self.__class__(super().__or__(other))

    def __and__(self, other):
        """
        Intersection is the minimum of corresponding counts.

        >>> FreqDist('abbb') & FreqDist('bcc')
        FreqDist({'b': 1})

        """
        return self.__class__(super().__and__(other))

    def __le__(self, other):
        """
        Returns True if this frequency distribution is a subset of the other
        and for no key the value exceeds the value of the same key from
        the other frequency distribution.

        The <= operator forms partial order and satisfying the axioms
        reflexivity, antisymmetry and transitivity.

        >>> FreqDist('a') <= FreqDist('a')
        True
        >>> a = FreqDist('abc')
        >>> b = FreqDist('aabc')
        >>> (a <= b, b <= a)
        (True, False)
        >>> FreqDist('a') <= FreqDist('abcd')
        True
        >>> FreqDist('abc') <= FreqDist('xyz')
        False
        >>> FreqDist('xyz') <= FreqDist('abc')
        False
        >>> c = FreqDist('a')
        >>> d = FreqDist('aa')
        >>> e = FreqDist('aaa')
        >>> c <= d and d <= e and c <= e
        True
        """
        if not isinstance(other, FreqDist):
            raise_unorderable_types('<=', self, other)
        return set(self).issubset(other) and all((self[key] <= other[key] for key in self))

    def __ge__(self, other):
        if not isinstance(other, FreqDist):
            raise_unorderable_types('>=', self, other)
        return set(self).issuperset(other) and all((self[key] >= other[key] for key in other))
    __lt__ = lambda self, other: self <= other and (not self == other)
    __gt__ = lambda self, other: self >= other and (not self == other)

    def __repr__(self):
        """
        Return a string representation of this FreqDist.

        :rtype: string
        """
        return self.pformat()

    def pprint(self, maxlen=10, stream=None):
        """
        Print a string representation of this FreqDist to 'stream'

        :param maxlen: The maximum number of items to print
        :type maxlen: int
        :param stream: The stream to print to. stdout by default
        """
        print(self.pformat(maxlen=maxlen), file=stream)

    def pformat(self, maxlen=10):
        """
        Return a string representation of this FreqDist.

        :param maxlen: The maximum number of items to display
        :type maxlen: int
        :rtype: string
        """
        items = ['{!r}: {!r}'.format(*item) for item in self.most_common(maxlen)]
        if len(self) > maxlen:
            items.append('...')
        return 'FreqDist({{{0}}})'.format(', '.join(items))

    def __str__(self):
        """
        Return a string representation of this FreqDist.

        :rtype: string
        """
        return '<FreqDist with %d samples and %d outcomes>' % (len(self), self.N())

    def __iter__(self):
        """
        Return an iterator which yields tokens ordered by frequency.

        :rtype: iterator
        """
        for token, _ in self.most_common(self.B()):
            yield token