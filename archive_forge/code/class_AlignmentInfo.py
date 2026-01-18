from bisect import insort_left
from collections import defaultdict
from copy import deepcopy
from math import ceil
class AlignmentInfo:
    """
    Helper data object for training IBM Models 3 and up

    Read-only. For a source sentence and its counterpart in the target
    language, this class holds information about the sentence pair's
    alignment, cepts, and fertility.

    Warning: Alignments are one-indexed here, in contrast to
    nltk.translate.Alignment and AlignedSent, which are zero-indexed
    This class is not meant to be used outside of IBM models.
    """

    def __init__(self, alignment, src_sentence, trg_sentence, cepts):
        if not isinstance(alignment, tuple):
            raise TypeError('The alignment must be a tuple because it is used to uniquely identify AlignmentInfo objects.')
        self.alignment = alignment
        '\n        tuple(int): Alignment function. ``alignment[j]`` is the position\n        in the source sentence that is aligned to the position j in the\n        target sentence.\n        '
        self.src_sentence = src_sentence
        '\n        tuple(str): Source sentence referred to by this object.\n        Should include NULL token (None) in index 0.\n        '
        self.trg_sentence = trg_sentence
        '\n        tuple(str): Target sentence referred to by this object.\n        Should have a dummy element in index 0 so that the first word\n        starts from index 1.\n        '
        self.cepts = cepts
        '\n        list(list(int)): The positions of the target words, in\n        ascending order, aligned to a source word position. For example,\n        cepts[4] = (2, 3, 7) means that words in positions 2, 3 and 7\n        of the target sentence are aligned to the word in position 4 of\n        the source sentence\n        '
        self.score = None
        '\n        float: Optional. Probability of alignment, as defined by the\n        IBM model that assesses this alignment\n        '

    def fertility_of_i(self, i):
        """
        Fertility of word in position ``i`` of the source sentence
        """
        return len(self.cepts[i])

    def is_head_word(self, j):
        """
        :return: Whether the word in position ``j`` of the target
            sentence is a head word
        """
        i = self.alignment[j]
        return self.cepts[i][0] == j

    def center_of_cept(self, i):
        """
        :return: The ceiling of the average positions of the words in
            the tablet of cept ``i``, or 0 if ``i`` is None
        """
        if i is None:
            return 0
        average_position = sum(self.cepts[i]) / len(self.cepts[i])
        return int(ceil(average_position))

    def previous_cept(self, j):
        """
        :return: The previous cept of ``j``, or None if ``j`` belongs to
            the first cept
        """
        i = self.alignment[j]
        if i == 0:
            raise ValueError('Words aligned to NULL cannot have a previous cept because NULL has no position')
        previous_cept = i - 1
        while previous_cept > 0 and self.fertility_of_i(previous_cept) == 0:
            previous_cept -= 1
        if previous_cept <= 0:
            previous_cept = None
        return previous_cept

    def previous_in_tablet(self, j):
        """
        :return: The position of the previous word that is in the same
            tablet as ``j``, or None if ``j`` is the first word of the
            tablet
        """
        i = self.alignment[j]
        tablet_position = self.cepts[i].index(j)
        if tablet_position == 0:
            return None
        return self.cepts[i][tablet_position - 1]

    def zero_indexed_alignment(self):
        """
        :return: Zero-indexed alignment, suitable for use in external
            ``nltk.translate`` modules like ``nltk.translate.Alignment``
        :rtype: list(tuple)
        """
        zero_indexed_alignment = []
        for j in range(1, len(self.trg_sentence)):
            i = self.alignment[j] - 1
            if i < 0:
                i = None
            zero_indexed_alignment.append((j - 1, i))
        return zero_indexed_alignment

    def __eq__(self, other):
        return self.alignment == other.alignment

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.alignment)