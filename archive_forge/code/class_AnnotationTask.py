import logging
from itertools import groupby
from operator import itemgetter
from nltk.internals import deprecated
from nltk.metrics.distance import binary_distance
from nltk.probability import ConditionalFreqDist, FreqDist
class AnnotationTask:
    """Represents an annotation task, i.e. people assign labels to items.

    Notation tries to match notation in Artstein and Poesio (2007).

    In general, coders and items can be represented as any hashable object.
    Integers, for example, are fine, though strings are more readable.
    Labels must support the distance functions applied to them, so e.g.
    a string-edit-distance makes no sense if your labels are integers,
    whereas interval distance needs numeric values.  A notable case of this
    is the MASI metric, which requires Python sets.
    """

    def __init__(self, data=None, distance=binary_distance):
        """Initialize an annotation task.

        The data argument can be None (to create an empty annotation task) or a sequence of 3-tuples,
        each representing a coder's labeling of an item:
        ``(coder,item,label)``

        The distance argument is a function taking two arguments (labels) and producing a numerical distance.
        The distance from a label to itself should be zero:
        ``distance(l,l) = 0``
        """
        self.distance = distance
        self.I = set()
        self.K = set()
        self.C = set()
        self.data = []
        if data is not None:
            self.load_array(data)

    def __str__(self):
        return '\r\n'.join(map(lambda x: '%s\t%s\t%s' % (x['coder'], x['item'].replace('_', '\t'), ','.join(x['labels'])), self.data))

    def load_array(self, array):
        """Load an sequence of annotation results, appending to any data already loaded.

        The argument is a sequence of 3-tuples, each representing a coder's labeling of an item:
            (coder,item,label)
        """
        for coder, item, labels in array:
            self.C.add(coder)
            self.K.add(labels)
            self.I.add(item)
            self.data.append({'coder': coder, 'labels': labels, 'item': item})

    def agr(self, cA, cB, i, data=None):
        """Agreement between two coders on a given item"""
        data = data or self.data
        k1 = next((x for x in data if x['coder'] in (cA, cB) and x['item'] == i))
        if k1['coder'] == cA:
            k2 = next((x for x in data if x['coder'] == cB and x['item'] == i))
        else:
            k2 = next((x for x in data if x['coder'] == cA and x['item'] == i))
        ret = 1.0 - float(self.distance(k1['labels'], k2['labels']))
        log.debug('Observed agreement between %s and %s on %s: %f', cA, cB, i, ret)
        log.debug('Distance between "%r" and "%r": %f', k1['labels'], k2['labels'], 1.0 - ret)
        return ret

    def Nk(self, k):
        return float(sum((1 for x in self.data if x['labels'] == k)))

    def Nik(self, i, k):
        return float(sum((1 for x in self.data if x['item'] == i and x['labels'] == k)))

    def Nck(self, c, k):
        return float(sum((1 for x in self.data if x['coder'] == c and x['labels'] == k)))

    @deprecated('Use Nk, Nik or Nck instead')
    def N(self, k=None, i=None, c=None):
        """Implements the "n-notation" used in Artstein and Poesio (2007)"""
        if k is not None and i is None and (c is None):
            ret = self.Nk(k)
        elif k is not None and i is not None and (c is None):
            ret = self.Nik(i, k)
        elif k is not None and c is not None and (i is None):
            ret = self.Nck(c, k)
        else:
            raise ValueError(f'You must pass either i or c, not both! (k={k!r},i={i!r},c={c!r})')
        log.debug('Count on N[%s,%s,%s]: %d', k, i, c, ret)
        return ret

    def _grouped_data(self, field, data=None):
        data = data or self.data
        return groupby(sorted(data, key=itemgetter(field)), itemgetter(field))

    def Ao(self, cA, cB):
        """Observed agreement between two coders on all items."""
        data = self._grouped_data('item', (x for x in self.data if x['coder'] in (cA, cB)))
        ret = sum((self.agr(cA, cB, item, item_data) for item, item_data in data)) / len(self.I)
        log.debug('Observed agreement between %s and %s: %f', cA, cB, ret)
        return ret

    def _pairwise_average(self, function):
        """
        Calculates the average of function results for each coder pair
        """
        total = 0
        n = 0
        s = self.C.copy()
        for cA in self.C:
            s.remove(cA)
            for cB in s:
                total += function(cA, cB)
                n += 1
        ret = total / n
        return ret

    def avg_Ao(self):
        """Average observed agreement across all coders and items."""
        ret = self._pairwise_average(self.Ao)
        log.debug('Average observed agreement: %f', ret)
        return ret

    def Do_Kw_pairwise(self, cA, cB, max_distance=1.0):
        """The observed disagreement for the weighted kappa coefficient."""
        total = 0.0
        data = (x for x in self.data if x['coder'] in (cA, cB))
        for i, itemdata in self._grouped_data('item', data):
            total += self.distance(next(itemdata)['labels'], next(itemdata)['labels'])
        ret = total / (len(self.I) * max_distance)
        log.debug('Observed disagreement between %s and %s: %f', cA, cB, ret)
        return ret

    def Do_Kw(self, max_distance=1.0):
        """Averaged over all labelers"""
        ret = self._pairwise_average(lambda cA, cB: self.Do_Kw_pairwise(cA, cB, max_distance))
        log.debug('Observed disagreement: %f', ret)
        return ret

    def S(self):
        """Bennett, Albert and Goldstein 1954"""
        Ae = 1.0 / len(self.K)
        ret = (self.avg_Ao() - Ae) / (1.0 - Ae)
        return ret

    def pi(self):
        """Scott 1955; here, multi-pi.
        Equivalent to K from Siegel and Castellan (1988).

        """
        total = 0.0
        label_freqs = FreqDist((x['labels'] for x in self.data))
        for k, f in label_freqs.items():
            total += f ** 2
        Ae = total / (len(self.I) * len(self.C)) ** 2
        return (self.avg_Ao() - Ae) / (1 - Ae)

    def Ae_kappa(self, cA, cB):
        Ae = 0.0
        nitems = float(len(self.I))
        label_freqs = ConditionalFreqDist(((x['labels'], x['coder']) for x in self.data))
        for k in label_freqs.conditions():
            Ae += label_freqs[k][cA] / nitems * (label_freqs[k][cB] / nitems)
        return Ae

    def kappa_pairwise(self, cA, cB):
        """ """
        Ae = self.Ae_kappa(cA, cB)
        ret = (self.Ao(cA, cB) - Ae) / (1.0 - Ae)
        log.debug('Expected agreement between %s and %s: %f', cA, cB, Ae)
        return ret

    def kappa(self):
        """Cohen 1960
        Averages naively over kappas for each coder pair.

        """
        return self._pairwise_average(self.kappa_pairwise)

    def multi_kappa(self):
        """Davies and Fleiss 1982
        Averages over observed and expected agreements for each coder pair.

        """
        Ae = self._pairwise_average(self.Ae_kappa)
        return (self.avg_Ao() - Ae) / (1.0 - Ae)

    def Disagreement(self, label_freqs):
        total_labels = sum(label_freqs.values())
        pairs = 0.0
        for j, nj in label_freqs.items():
            for l, nl in label_freqs.items():
                pairs += float(nj * nl) * self.distance(l, j)
        return 1.0 * pairs / (total_labels * (total_labels - 1))

    def alpha(self):
        """Krippendorff 1980"""
        if len(self.K) == 0:
            raise ValueError('Cannot calculate alpha, no data present!')
        if len(self.K) == 1:
            log.debug('Only one annotation value, alpha returning 1.')
            return 1
        if len(self.C) == 1 and len(self.I) == 1:
            raise ValueError('Cannot calculate alpha, only one coder and item present!')
        total_disagreement = 0.0
        total_ratings = 0
        all_valid_labels_freq = FreqDist([])
        total_do = 0.0
        for i, itemdata in self._grouped_data('item'):
            label_freqs = FreqDist((x['labels'] for x in itemdata))
            labels_count = sum(label_freqs.values())
            if labels_count < 2:
                continue
            all_valid_labels_freq += label_freqs
            total_do += self.Disagreement(label_freqs) * labels_count
        do = total_do / sum(all_valid_labels_freq.values())
        de = self.Disagreement(all_valid_labels_freq)
        k_alpha = 1.0 - do / de
        return k_alpha

    def weighted_kappa_pairwise(self, cA, cB, max_distance=1.0):
        """Cohen 1968"""
        total = 0.0
        label_freqs = ConditionalFreqDist(((x['coder'], x['labels']) for x in self.data if x['coder'] in (cA, cB)))
        for j in self.K:
            for l in self.K:
                total += label_freqs[cA][j] * label_freqs[cB][l] * self.distance(j, l)
        De = total / (max_distance * pow(len(self.I), 2))
        log.debug('Expected disagreement between %s and %s: %f', cA, cB, De)
        Do = self.Do_Kw_pairwise(cA, cB)
        ret = 1.0 - Do / De
        return ret

    def weighted_kappa(self, max_distance=1.0):
        """Cohen 1968"""
        return self._pairwise_average(lambda cA, cB: self.weighted_kappa_pairwise(cA, cB, max_distance))