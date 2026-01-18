import warnings
from collections import defaultdict
from nltk.translate import AlignedSent, Alignment, IBMModel
from nltk.translate.ibm_model import Counts
class IBMModel1(IBMModel):
    """
    Lexical translation model that ignores word order

    >>> bitext = []
    >>> bitext.append(AlignedSent(['klein', 'ist', 'das', 'haus'], ['the', 'house', 'is', 'small']))
    >>> bitext.append(AlignedSent(['das', 'haus', 'ist', 'ja', 'groß'], ['the', 'house', 'is', 'big']))
    >>> bitext.append(AlignedSent(['das', 'buch', 'ist', 'ja', 'klein'], ['the', 'book', 'is', 'small']))
    >>> bitext.append(AlignedSent(['das', 'haus'], ['the', 'house']))
    >>> bitext.append(AlignedSent(['das', 'buch'], ['the', 'book']))
    >>> bitext.append(AlignedSent(['ein', 'buch'], ['a', 'book']))

    >>> ibm1 = IBMModel1(bitext, 5)

    >>> print(round(ibm1.translation_table['buch']['book'], 3))
    0.889
    >>> print(round(ibm1.translation_table['das']['book'], 3))
    0.062
    >>> print(round(ibm1.translation_table['buch'][None], 3))
    0.113
    >>> print(round(ibm1.translation_table['ja'][None], 3))
    0.073

    >>> test_sentence = bitext[2]
    >>> test_sentence.words
    ['das', 'buch', 'ist', 'ja', 'klein']
    >>> test_sentence.mots
    ['the', 'book', 'is', 'small']
    >>> test_sentence.alignment
    Alignment([(0, 0), (1, 1), (2, 2), (3, 2), (4, 3)])

    """

    def __init__(self, sentence_aligned_corpus, iterations, probability_tables=None):
        """
        Train on ``sentence_aligned_corpus`` and create a lexical
        translation model.

        Translation direction is from ``AlignedSent.mots`` to
        ``AlignedSent.words``.

        :param sentence_aligned_corpus: Sentence-aligned parallel corpus
        :type sentence_aligned_corpus: list(AlignedSent)

        :param iterations: Number of iterations to run training algorithm
        :type iterations: int

        :param probability_tables: Optional. Use this to pass in custom
            probability values. If not specified, probabilities will be
            set to a uniform distribution, or some other sensible value.
            If specified, the following entry must be present:
            ``translation_table``.
            See ``IBMModel`` for the type and purpose of this table.
        :type probability_tables: dict[str]: object
        """
        super().__init__(sentence_aligned_corpus)
        if probability_tables is None:
            self.set_uniform_probabilities(sentence_aligned_corpus)
        else:
            self.translation_table = probability_tables['translation_table']
        for n in range(0, iterations):
            self.train(sentence_aligned_corpus)
        self.align_all(sentence_aligned_corpus)

    def set_uniform_probabilities(self, sentence_aligned_corpus):
        initial_prob = 1 / len(self.trg_vocab)
        if initial_prob < IBMModel.MIN_PROB:
            warnings.warn('Target language vocabulary is too large (' + str(len(self.trg_vocab)) + ' words). Results may be less accurate.')
        for t in self.trg_vocab:
            self.translation_table[t] = defaultdict(lambda: initial_prob)

    def train(self, parallel_corpus):
        counts = Counts()
        for aligned_sentence in parallel_corpus:
            trg_sentence = aligned_sentence.words
            src_sentence = [None] + aligned_sentence.mots
            total_count = self.prob_all_alignments(src_sentence, trg_sentence)
            for t in trg_sentence:
                for s in src_sentence:
                    count = self.prob_alignment_point(s, t)
                    normalized_count = count / total_count[t]
                    counts.t_given_s[t][s] += normalized_count
                    counts.any_t_given_s[s] += normalized_count
        self.maximize_lexical_translation_probabilities(counts)

    def prob_all_alignments(self, src_sentence, trg_sentence):
        """
        Computes the probability of all possible word alignments,
        expressed as a marginal distribution over target words t

        Each entry in the return value represents the contribution to
        the total alignment probability by the target word t.

        To obtain probability(alignment | src_sentence, trg_sentence),
        simply sum the entries in the return value.

        :return: Probability of t for all s in ``src_sentence``
        :rtype: dict(str): float
        """
        alignment_prob_for_t = defaultdict(lambda: 0.0)
        for t in trg_sentence:
            for s in src_sentence:
                alignment_prob_for_t[t] += self.prob_alignment_point(s, t)
        return alignment_prob_for_t

    def prob_alignment_point(self, s, t):
        """
        Probability that word ``t`` in the target sentence is aligned to
        word ``s`` in the source sentence
        """
        return self.translation_table[t][s]

    def prob_t_a_given_s(self, alignment_info):
        """
        Probability of target sentence and an alignment given the
        source sentence
        """
        prob = 1.0
        for j, i in enumerate(alignment_info.alignment):
            if j == 0:
                continue
            trg_word = alignment_info.trg_sentence[j]
            src_word = alignment_info.src_sentence[i]
            prob *= self.translation_table[trg_word][src_word]
        return max(prob, IBMModel.MIN_PROB)

    def align_all(self, parallel_corpus):
        for sentence_pair in parallel_corpus:
            self.align(sentence_pair)

    def align(self, sentence_pair):
        """
        Determines the best word alignment for one sentence pair from
        the corpus that the model was trained on.

        The best alignment will be set in ``sentence_pair`` when the
        method returns. In contrast with the internal implementation of
        IBM models, the word indices in the ``Alignment`` are zero-
        indexed, not one-indexed.

        :param sentence_pair: A sentence in the source language and its
            counterpart sentence in the target language
        :type sentence_pair: AlignedSent
        """
        best_alignment = []
        for j, trg_word in enumerate(sentence_pair.words):
            best_prob = max(self.translation_table[trg_word][None], IBMModel.MIN_PROB)
            best_alignment_point = None
            for i, src_word in enumerate(sentence_pair.mots):
                align_prob = self.translation_table[trg_word][src_word]
                if align_prob >= best_prob:
                    best_prob = align_prob
                    best_alignment_point = i
            best_alignment.append((j, best_alignment_point))
        sentence_pair.alignment = Alignment(best_alignment)