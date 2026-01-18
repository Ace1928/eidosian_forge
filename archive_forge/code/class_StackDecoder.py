import warnings
from collections import defaultdict
from math import log
class StackDecoder:
    """
    Phrase-based stack decoder for machine translation

    >>> from nltk.translate import PhraseTable
    >>> phrase_table = PhraseTable()
    >>> phrase_table.add(('niemand',), ('nobody',), log(0.8))
    >>> phrase_table.add(('niemand',), ('no', 'one'), log(0.2))
    >>> phrase_table.add(('erwartet',), ('expects',), log(0.8))
    >>> phrase_table.add(('erwartet',), ('expecting',), log(0.2))
    >>> phrase_table.add(('niemand', 'erwartet'), ('one', 'does', 'not', 'expect'), log(0.1))
    >>> phrase_table.add(('die', 'spanische', 'inquisition'), ('the', 'spanish', 'inquisition'), log(0.8))
    >>> phrase_table.add(('!',), ('!',), log(0.8))

    >>> #  nltk.model should be used here once it is implemented
    >>> from collections import defaultdict
    >>> language_prob = defaultdict(lambda: -999.0)
    >>> language_prob[('nobody',)] = log(0.5)
    >>> language_prob[('expects',)] = log(0.4)
    >>> language_prob[('the', 'spanish', 'inquisition')] = log(0.2)
    >>> language_prob[('!',)] = log(0.1)
    >>> language_model = type('',(object,),{'probability_change': lambda self, context, phrase: language_prob[phrase], 'probability': lambda self, phrase: language_prob[phrase]})()

    >>> stack_decoder = StackDecoder(phrase_table, language_model)

    >>> stack_decoder.translate(['niemand', 'erwartet', 'die', 'spanische', 'inquisition', '!'])
    ['nobody', 'expects', 'the', 'spanish', 'inquisition', '!']

    """

    def __init__(self, phrase_table, language_model):
        """
        :param phrase_table: Table of translations for source language
            phrases and the log probabilities for those translations.
        :type phrase_table: PhraseTable

        :param language_model: Target language model. Must define a
            ``probability_change`` method that calculates the change in
            log probability of a sentence, if a given string is appended
            to it.
            This interface is experimental and will likely be replaced
            with nltk.model once it is implemented.
        :type language_model: object
        """
        self.phrase_table = phrase_table
        self.language_model = language_model
        self.word_penalty = 0.0
        '\n        float: Influences the translation length exponentially.\n            If positive, shorter translations are preferred.\n            If negative, longer translations are preferred.\n            If zero, no penalty is applied.\n        '
        self.beam_threshold = 0.0
        '\n        float: Hypotheses that score below this factor of the best\n            hypothesis in a stack are dropped from consideration.\n            Value between 0.0 and 1.0.\n        '
        self.stack_size = 100
        '\n        int: Maximum number of hypotheses to consider in a stack.\n            Higher values increase the likelihood of a good translation,\n            but increases processing time.\n        '
        self.__distortion_factor = 0.5
        self.__compute_log_distortion()

    @property
    def distortion_factor(self):
        """
        float: Amount of reordering of source phrases.
            Lower values favour monotone translation, suitable when
            word order is similar for both source and target languages.
            Value between 0.0 and 1.0. Default 0.5.
        """
        return self.__distortion_factor

    @distortion_factor.setter
    def distortion_factor(self, d):
        self.__distortion_factor = d
        self.__compute_log_distortion()

    def __compute_log_distortion(self):
        if self.__distortion_factor == 0.0:
            self.__log_distortion_factor = log(1e-09)
        else:
            self.__log_distortion_factor = log(self.__distortion_factor)

    def translate(self, src_sentence):
        """
        :param src_sentence: Sentence to be translated
        :type src_sentence: list(str)

        :return: Translated sentence
        :rtype: list(str)
        """
        sentence = tuple(src_sentence)
        sentence_length = len(sentence)
        stacks = [_Stack(self.stack_size, self.beam_threshold) for _ in range(0, sentence_length + 1)]
        empty_hypothesis = _Hypothesis()
        stacks[0].push(empty_hypothesis)
        all_phrases = self.find_all_src_phrases(sentence)
        future_score_table = self.compute_future_scores(sentence)
        for stack in stacks:
            for hypothesis in stack:
                possible_expansions = StackDecoder.valid_phrases(all_phrases, hypothesis)
                for src_phrase_span in possible_expansions:
                    src_phrase = sentence[src_phrase_span[0]:src_phrase_span[1]]
                    for translation_option in self.phrase_table.translations_for(src_phrase):
                        raw_score = self.expansion_score(hypothesis, translation_option, src_phrase_span)
                        new_hypothesis = _Hypothesis(raw_score=raw_score, src_phrase_span=src_phrase_span, trg_phrase=translation_option.trg_phrase, previous=hypothesis)
                        new_hypothesis.future_score = self.future_score(new_hypothesis, future_score_table, sentence_length)
                        total_words = new_hypothesis.total_translated_words()
                        stacks[total_words].push(new_hypothesis)
        if not stacks[sentence_length]:
            warnings.warn('Unable to translate all words. The source sentence contains words not in the phrase table')
            return []
        best_hypothesis = stacks[sentence_length].best()
        return best_hypothesis.translation_so_far()

    def find_all_src_phrases(self, src_sentence):
        """
        Finds all subsequences in src_sentence that have a phrase
        translation in the translation table

        :type src_sentence: tuple(str)

        :return: Subsequences that have a phrase translation,
            represented as a table of lists of end positions.
            For example, if result[2] is [5, 6, 9], then there are
            three phrases starting from position 2 in ``src_sentence``,
            ending at positions 5, 6, and 9 exclusive. The list of
            ending positions are in ascending order.
        :rtype: list(list(int))
        """
        sentence_length = len(src_sentence)
        phrase_indices = [[] for _ in src_sentence]
        for start in range(0, sentence_length):
            for end in range(start + 1, sentence_length + 1):
                potential_phrase = src_sentence[start:end]
                if potential_phrase in self.phrase_table:
                    phrase_indices[start].append(end)
        return phrase_indices

    def compute_future_scores(self, src_sentence):
        """
        Determines the approximate scores for translating every
        subsequence in ``src_sentence``

        Future scores can be used a look-ahead to determine the
        difficulty of translating the remaining parts of a src_sentence.

        :type src_sentence: tuple(str)

        :return: Scores of subsequences referenced by their start and
            end positions. For example, result[2][5] is the score of the
            subsequence covering positions 2, 3, and 4.
        :rtype: dict(int: (dict(int): float))
        """
        scores = defaultdict(lambda: defaultdict(lambda: float('-inf')))
        for seq_length in range(1, len(src_sentence) + 1):
            for start in range(0, len(src_sentence) - seq_length + 1):
                end = start + seq_length
                phrase = src_sentence[start:end]
                if phrase in self.phrase_table:
                    score = self.phrase_table.translations_for(phrase)[0].log_prob
                    score += self.language_model.probability(phrase)
                    scores[start][end] = score
                for mid in range(start + 1, end):
                    combined_score = scores[start][mid] + scores[mid][end]
                    if combined_score > scores[start][end]:
                        scores[start][end] = combined_score
        return scores

    def future_score(self, hypothesis, future_score_table, sentence_length):
        """
        Determines the approximate score for translating the
        untranslated words in ``hypothesis``
        """
        score = 0.0
        for span in hypothesis.untranslated_spans(sentence_length):
            score += future_score_table[span[0]][span[1]]
        return score

    def expansion_score(self, hypothesis, translation_option, src_phrase_span):
        """
        Calculate the score of expanding ``hypothesis`` with
        ``translation_option``

        :param hypothesis: Hypothesis being expanded
        :type hypothesis: _Hypothesis

        :param translation_option: Information about the proposed expansion
        :type translation_option: PhraseTableEntry

        :param src_phrase_span: Word position span of the source phrase
        :type src_phrase_span: tuple(int, int)
        """
        score = hypothesis.raw_score
        score += translation_option.log_prob
        score += self.language_model.probability_change(hypothesis, translation_option.trg_phrase)
        score += self.distortion_score(hypothesis, src_phrase_span)
        score -= self.word_penalty * len(translation_option.trg_phrase)
        return score

    def distortion_score(self, hypothesis, next_src_phrase_span):
        if not hypothesis.src_phrase_span:
            return 0.0
        next_src_phrase_start = next_src_phrase_span[0]
        prev_src_phrase_end = hypothesis.src_phrase_span[1]
        distortion_distance = next_src_phrase_start - prev_src_phrase_end
        return abs(distortion_distance) * self.__log_distortion_factor

    @staticmethod
    def valid_phrases(all_phrases_from, hypothesis):
        """
        Extract phrases from ``all_phrases_from`` that contains words
        that have not been translated by ``hypothesis``

        :param all_phrases_from: Phrases represented by their spans, in
            the same format as the return value of
            ``find_all_src_phrases``
        :type all_phrases_from: list(list(int))

        :type hypothesis: _Hypothesis

        :return: A list of phrases, represented by their spans, that
            cover untranslated positions.
        :rtype: list(tuple(int, int))
        """
        untranslated_spans = hypothesis.untranslated_spans(len(all_phrases_from))
        valid_phrases = []
        for available_span in untranslated_spans:
            start = available_span[0]
            available_end = available_span[1]
            while start < available_end:
                for phrase_end in all_phrases_from[start]:
                    if phrase_end > available_end:
                        break
                    valid_phrases.append((start, phrase_end))
                start += 1
        return valid_phrases