from bisect import insort_left
from collections import defaultdict
from copy import deepcopy
from math import ceil
class IBMModel:
    """
    Abstract base class for all IBM models
    """
    MIN_PROB = 1e-12

    def __init__(self, sentence_aligned_corpus):
        self.init_vocab(sentence_aligned_corpus)
        self.reset_probabilities()

    def reset_probabilities(self):
        self.translation_table = defaultdict(lambda: defaultdict(lambda: IBMModel.MIN_PROB))
        '\n        dict[str][str]: float. Probability(target word | source word).\n        Values accessed as ``translation_table[target_word][source_word]``.\n        '
        self.alignment_table = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: IBMModel.MIN_PROB))))
        '\n        dict[int][int][int][int]: float. Probability(i | j,l,m).\n        Values accessed as ``alignment_table[i][j][l][m]``.\n        Used in model 2 and hill climbing in models 3 and above\n        '
        self.fertility_table = defaultdict(lambda: defaultdict(lambda: self.MIN_PROB))
        '\n        dict[int][str]: float. Probability(fertility | source word).\n        Values accessed as ``fertility_table[fertility][source_word]``.\n        Used in model 3 and higher.\n        '
        self.p1 = 0.5
        '\n        Probability that a generated word requires another target word\n        that is aligned to NULL.\n        Used in model 3 and higher.\n        '

    def set_uniform_probabilities(self, sentence_aligned_corpus):
        """
        Initialize probability tables to a uniform distribution

        Derived classes should implement this accordingly.
        """
        pass

    def init_vocab(self, sentence_aligned_corpus):
        src_vocab = set()
        trg_vocab = set()
        for aligned_sentence in sentence_aligned_corpus:
            trg_vocab.update(aligned_sentence.words)
            src_vocab.update(aligned_sentence.mots)
        src_vocab.add(None)
        self.src_vocab = src_vocab
        '\n        set(str): All source language words used in training\n        '
        self.trg_vocab = trg_vocab
        '\n        set(str): All target language words used in training\n        '

    def sample(self, sentence_pair):
        """
        Sample the most probable alignments from the entire alignment
        space

        First, determine the best alignment according to IBM Model 2.
        With this initial alignment, use hill climbing to determine the
        best alignment according to a higher IBM Model. Add this
        alignment and its neighbors to the sample set. Repeat this
        process with other initial alignments obtained by pegging an
        alignment point.

        Hill climbing may be stuck in a local maxima, hence the pegging
        and trying out of different alignments.

        :param sentence_pair: Source and target language sentence pair
            to generate a sample of alignments from
        :type sentence_pair: AlignedSent

        :return: A set of best alignments represented by their ``AlignmentInfo``
            and the best alignment of the set for convenience
        :rtype: set(AlignmentInfo), AlignmentInfo
        """
        sampled_alignments = set()
        l = len(sentence_pair.mots)
        m = len(sentence_pair.words)
        initial_alignment = self.best_model2_alignment(sentence_pair)
        potential_alignment = self.hillclimb(initial_alignment)
        sampled_alignments.update(self.neighboring(potential_alignment))
        best_alignment = potential_alignment
        for j in range(1, m + 1):
            for i in range(0, l + 1):
                initial_alignment = self.best_model2_alignment(sentence_pair, j, i)
                potential_alignment = self.hillclimb(initial_alignment, j)
                neighbors = self.neighboring(potential_alignment, j)
                sampled_alignments.update(neighbors)
                if potential_alignment.score > best_alignment.score:
                    best_alignment = potential_alignment
        return (sampled_alignments, best_alignment)

    def best_model2_alignment(self, sentence_pair, j_pegged=None, i_pegged=0):
        """
        Finds the best alignment according to IBM Model 2

        Used as a starting point for hill climbing in Models 3 and
        above, because it is easier to compute than the best alignments
        in higher models

        :param sentence_pair: Source and target language sentence pair
            to be word-aligned
        :type sentence_pair: AlignedSent

        :param j_pegged: If specified, the alignment point of j_pegged
            will be fixed to i_pegged
        :type j_pegged: int

        :param i_pegged: Alignment point to j_pegged
        :type i_pegged: int
        """
        src_sentence = [None] + sentence_pair.mots
        trg_sentence = ['UNUSED'] + sentence_pair.words
        l = len(src_sentence) - 1
        m = len(trg_sentence) - 1
        alignment = [0] * (m + 1)
        cepts = [[] for i in range(l + 1)]
        for j in range(1, m + 1):
            if j == j_pegged:
                best_i = i_pegged
            else:
                best_i = 0
                max_alignment_prob = IBMModel.MIN_PROB
                t = trg_sentence[j]
                for i in range(0, l + 1):
                    s = src_sentence[i]
                    alignment_prob = self.translation_table[t][s] * self.alignment_table[i][j][l][m]
                    if alignment_prob >= max_alignment_prob:
                        max_alignment_prob = alignment_prob
                        best_i = i
            alignment[j] = best_i
            cepts[best_i].append(j)
        return AlignmentInfo(tuple(alignment), tuple(src_sentence), tuple(trg_sentence), cepts)

    def hillclimb(self, alignment_info, j_pegged=None):
        """
        Starting from the alignment in ``alignment_info``, look at
        neighboring alignments iteratively for the best one

        There is no guarantee that the best alignment in the alignment
        space will be found, because the algorithm might be stuck in a
        local maximum.

        :param j_pegged: If specified, the search will be constrained to
            alignments where ``j_pegged`` remains unchanged
        :type j_pegged: int

        :return: The best alignment found from hill climbing
        :rtype: AlignmentInfo
        """
        alignment = alignment_info
        max_probability = self.prob_t_a_given_s(alignment)
        while True:
            old_alignment = alignment
            for neighbor_alignment in self.neighboring(alignment, j_pegged):
                neighbor_probability = self.prob_t_a_given_s(neighbor_alignment)
                if neighbor_probability > max_probability:
                    alignment = neighbor_alignment
                    max_probability = neighbor_probability
            if alignment == old_alignment:
                break
        alignment.score = max_probability
        return alignment

    def neighboring(self, alignment_info, j_pegged=None):
        """
        Determine the neighbors of ``alignment_info``, obtained by
        moving or swapping one alignment point

        :param j_pegged: If specified, neighbors that have a different
            alignment point from j_pegged will not be considered
        :type j_pegged: int

        :return: A set neighboring alignments represented by their
            ``AlignmentInfo``
        :rtype: set(AlignmentInfo)
        """
        neighbors = set()
        l = len(alignment_info.src_sentence) - 1
        m = len(alignment_info.trg_sentence) - 1
        original_alignment = alignment_info.alignment
        original_cepts = alignment_info.cepts
        for j in range(1, m + 1):
            if j != j_pegged:
                for i in range(0, l + 1):
                    new_alignment = list(original_alignment)
                    new_cepts = deepcopy(original_cepts)
                    old_i = original_alignment[j]
                    new_alignment[j] = i
                    insort_left(new_cepts[i], j)
                    new_cepts[old_i].remove(j)
                    new_alignment_info = AlignmentInfo(tuple(new_alignment), alignment_info.src_sentence, alignment_info.trg_sentence, new_cepts)
                    neighbors.add(new_alignment_info)
        for j in range(1, m + 1):
            if j != j_pegged:
                for other_j in range(1, m + 1):
                    if other_j != j_pegged and other_j != j:
                        new_alignment = list(original_alignment)
                        new_cepts = deepcopy(original_cepts)
                        other_i = original_alignment[other_j]
                        i = original_alignment[j]
                        new_alignment[j] = other_i
                        new_alignment[other_j] = i
                        new_cepts[other_i].remove(other_j)
                        insort_left(new_cepts[other_i], j)
                        new_cepts[i].remove(j)
                        insort_left(new_cepts[i], other_j)
                        new_alignment_info = AlignmentInfo(tuple(new_alignment), alignment_info.src_sentence, alignment_info.trg_sentence, new_cepts)
                        neighbors.add(new_alignment_info)
        return neighbors

    def maximize_lexical_translation_probabilities(self, counts):
        for t, src_words in counts.t_given_s.items():
            for s in src_words:
                estimate = counts.t_given_s[t][s] / counts.any_t_given_s[s]
                self.translation_table[t][s] = max(estimate, IBMModel.MIN_PROB)

    def maximize_fertility_probabilities(self, counts):
        for phi, src_words in counts.fertility.items():
            for s in src_words:
                estimate = counts.fertility[phi][s] / counts.fertility_for_any_phi[s]
                self.fertility_table[phi][s] = max(estimate, IBMModel.MIN_PROB)

    def maximize_null_generation_probabilities(self, counts):
        p1_estimate = counts.p1 / (counts.p1 + counts.p0)
        p1_estimate = max(p1_estimate, IBMModel.MIN_PROB)
        self.p1 = min(p1_estimate, 1 - IBMModel.MIN_PROB)

    def prob_of_alignments(self, alignments):
        probability = 0
        for alignment_info in alignments:
            probability += self.prob_t_a_given_s(alignment_info)
        return probability

    def prob_t_a_given_s(self, alignment_info):
        """
        Probability of target sentence and an alignment given the
        source sentence

        All required information is assumed to be in ``alignment_info``
        and self.

        Derived classes should override this method
        """
        return 0.0