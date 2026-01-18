from math import sqrt
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