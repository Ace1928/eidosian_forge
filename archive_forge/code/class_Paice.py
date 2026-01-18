from math import sqrt
class Paice:
    """Class for storing lemmas, stems and evaluation metrics."""

    def __init__(self, lemmas, stems):
        """
        :param lemmas: A dictionary where keys are lemmas and values are sets
            or lists of words corresponding to that lemma.
        :param stems: A dictionary where keys are stems and values are sets
            or lists of words corresponding to that stem.
        :type lemmas: dict(str): list(str)
        :type stems: dict(str): set(str)
        """
        self.lemmas = lemmas
        self.stems = stems
        self.coords = []
        self.gumt, self.gdmt, self.gwmt, self.gdnt = (None, None, None, None)
        self.ui, self.oi, self.sw = (None, None, None)
        self.errt = None
        self.update()

    def __str__(self):
        text = ['Global Unachieved Merge Total (GUMT): %s\n' % self.gumt]
        text.append('Global Desired Merge Total (GDMT): %s\n' % self.gdmt)
        text.append('Global Wrongly-Merged Total (GWMT): %s\n' % self.gwmt)
        text.append('Global Desired Non-merge Total (GDNT): %s\n' % self.gdnt)
        text.append('Understemming Index (GUMT / GDMT): %s\n' % self.ui)
        text.append('Overstemming Index (GWMT / GDNT): %s\n' % self.oi)
        text.append('Stemming Weight (OI / UI): %s\n' % self.sw)
        text.append('Error-Rate Relative to Truncation (ERRT): %s\r\n' % self.errt)
        coordinates = ' '.join(['(%s, %s)' % item for item in self.coords])
        text.append('Truncation line: %s' % coordinates)
        return ''.join(text)

    def _get_truncation_indexes(self, words, cutlength):
        """Count (UI, OI) when stemming is done by truncating words at 'cutlength'.

        :param words: Words used for the analysis
        :param cutlength: Words are stemmed by cutting them at this length
        :type words: set(str) or list(str)
        :type cutlength: int
        :return: Understemming and overstemming indexes
        :rtype: tuple(int, int)
        """
        truncated = _truncate(words, cutlength)
        gumt, gdmt, gwmt, gdnt = _calculate(self.lemmas, truncated)
        ui, oi = _indexes(gumt, gdmt, gwmt, gdnt)[:2]
        return (ui, oi)

    def _get_truncation_coordinates(self, cutlength=0):
        """Count (UI, OI) pairs for truncation points until we find the segment where (ui, oi) crosses the truncation line.

        :param cutlength: Optional parameter to start counting from (ui, oi)
        coordinates gotten by stemming at this length. Useful for speeding up
        the calculations when you know the approximate location of the
        intersection.
        :type cutlength: int
        :return: List of coordinate pairs that define the truncation line
        :rtype: list(tuple(float, float))
        """
        words = get_words_from_dictionary(self.lemmas)
        maxlength = max((len(word) for word in words))
        coords = []
        while cutlength <= maxlength:
            pair = self._get_truncation_indexes(words, cutlength)
            if pair not in coords:
                coords.append(pair)
            if pair == (0.0, 0.0):
                return coords
            if len(coords) >= 2 and pair[0] > 0.0:
                derivative1 = _get_derivative(coords[-2])
                derivative2 = _get_derivative(coords[-1])
                if derivative1 >= self.sw >= derivative2:
                    return coords
            cutlength += 1
        return coords

    def _errt(self):
        """Count Error-Rate Relative to Truncation (ERRT).

        :return: ERRT, length of the line from origo to (UI, OI) divided by
        the length of the line from origo to the point defined by the same
        line when extended until the truncation line.
        :rtype: float
        """
        self.coords = self._get_truncation_coordinates()
        if (0.0, 0.0) in self.coords:
            if (self.ui, self.oi) != (0.0, 0.0):
                return float('inf')
            else:
                return float('nan')
        if (self.ui, self.oi) == (0.0, 0.0):
            return 0.0
        intersection = _count_intersection(((0, 0), (self.ui, self.oi)), self.coords[-2:])
        op = sqrt(self.ui ** 2 + self.oi ** 2)
        ot = sqrt(intersection[0] ** 2 + intersection[1] ** 2)
        return op / ot

    def update(self):
        """Update statistics after lemmas and stems have been set."""
        self.gumt, self.gdmt, self.gwmt, self.gdnt = _calculate(self.lemmas, self.stems)
        self.ui, self.oi, self.sw = _indexes(self.gumt, self.gdmt, self.gwmt, self.gdnt)
        self.errt = self._errt()