from math import sqrt
def _calculate_cut(lemmawords, stems):
    """Count understemmed and overstemmed pairs for (lemma, stem) pair with common words.

    :param lemmawords: Set or list of words corresponding to certain lemma.
    :param stems: A dictionary where keys are stems and values are sets
    or lists of words corresponding to that stem.
    :type lemmawords: set(str) or list(str)
    :type stems: dict(str): set(str)
    :return: Amount of understemmed and overstemmed pairs contributed by words
    existing in both lemmawords and stems.
    :rtype: tuple(float, float)
    """
    umt, wmt = (0.0, 0.0)
    for stem in stems:
        cut = set(lemmawords) & set(stems[stem])
        if cut:
            cutcount = len(cut)
            stemcount = len(stems[stem])
            umt += cutcount * (len(lemmawords) - cutcount)
            wmt += cutcount * (stemcount - cutcount)
    return (umt, wmt)