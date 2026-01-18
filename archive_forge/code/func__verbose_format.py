from abc import ABCMeta, abstractmethod
from nltk import jsontags
def _verbose_format(self):
    """
        Return a wordy, human-readable string representation
        of the given rule.

        Not sure how useful this is.
        """

    def condition_to_str(feature, value):
        return 'the {} of {} is "{}"'.format(feature.PROPERTY_NAME, range_to_str(feature.positions), value)

    def range_to_str(positions):
        if len(positions) == 1:
            p = positions[0]
            if p == 0:
                return 'this word'
            if p == -1:
                return 'the preceding word'
            elif p == 1:
                return 'the following word'
            elif p < 0:
                return 'word i-%d' % -p
            elif p > 0:
                return 'word i+%d' % p
        else:
            mx = max(positions)
            mn = min(positions)
            if mx - mn == len(positions) - 1:
                return 'words i%+d...i%+d' % (mn, mx)
            else:
                return 'words {{{}}}'.format(','.join(('i%+d' % d for d in positions)))
    replacement = f'{self.original_tag} -> {self.replacement_tag}'
    conditions = (' if ' if self._conditions else '') + ', and '.join((condition_to_str(f, v) for f, v in self._conditions))
    return replacement + conditions