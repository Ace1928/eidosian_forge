import unittest
from collections import defaultdict
from math import log
from nltk.translate import PhraseTable, StackDecoder
from nltk.translate.stack_decoder import _Hypothesis, _Stack
class TestStackDecoder(unittest.TestCase):

    def test_find_all_src_phrases(self):
        phrase_table = TestStackDecoder.create_fake_phrase_table()
        stack_decoder = StackDecoder(phrase_table, None)
        sentence = ('my', 'hovercraft', 'is', 'full', 'of', 'eels')
        src_phrase_spans = stack_decoder.find_all_src_phrases(sentence)
        self.assertEqual(src_phrase_spans[0], [2])
        self.assertEqual(src_phrase_spans[1], [2])
        self.assertEqual(src_phrase_spans[2], [3])
        self.assertEqual(src_phrase_spans[3], [5, 6])
        self.assertFalse(src_phrase_spans[4])
        self.assertEqual(src_phrase_spans[5], [6])

    def test_distortion_score(self):
        stack_decoder = StackDecoder(None, None)
        stack_decoder.distortion_factor = 0.5
        hypothesis = _Hypothesis()
        hypothesis.src_phrase_span = (3, 5)
        score = stack_decoder.distortion_score(hypothesis, (8, 10))
        expected_score = log(stack_decoder.distortion_factor) * (8 - 5)
        self.assertEqual(score, expected_score)

    def test_distortion_score_of_first_expansion(self):
        stack_decoder = StackDecoder(None, None)
        stack_decoder.distortion_factor = 0.5
        hypothesis = _Hypothesis()
        score = stack_decoder.distortion_score(hypothesis, (8, 10))
        self.assertEqual(score, 0.0)

    def test_compute_future_costs(self):
        phrase_table = TestStackDecoder.create_fake_phrase_table()
        language_model = TestStackDecoder.create_fake_language_model()
        stack_decoder = StackDecoder(phrase_table, language_model)
        sentence = ('my', 'hovercraft', 'is', 'full', 'of', 'eels')
        future_scores = stack_decoder.compute_future_scores(sentence)
        self.assertEqual(future_scores[1][2], phrase_table.translations_for(('hovercraft',))[0].log_prob + language_model.probability(('hovercraft',)))
        self.assertEqual(future_scores[0][2], phrase_table.translations_for(('my', 'hovercraft'))[0].log_prob + language_model.probability(('my', 'hovercraft')))

    def test_compute_future_costs_for_phrases_not_in_phrase_table(self):
        phrase_table = TestStackDecoder.create_fake_phrase_table()
        language_model = TestStackDecoder.create_fake_language_model()
        stack_decoder = StackDecoder(phrase_table, language_model)
        sentence = ('my', 'hovercraft', 'is', 'full', 'of', 'eels')
        future_scores = stack_decoder.compute_future_scores(sentence)
        self.assertEqual(future_scores[1][3], future_scores[1][2] + future_scores[2][3])

    def test_future_score(self):
        hypothesis = _Hypothesis()
        hypothesis.untranslated_spans = lambda _: [(0, 2), (5, 8)]
        future_score_table = defaultdict(lambda: defaultdict(float))
        future_score_table[0][2] = 0.4
        future_score_table[5][8] = 0.5
        stack_decoder = StackDecoder(None, None)
        future_score = stack_decoder.future_score(hypothesis, future_score_table, 8)
        self.assertEqual(future_score, 0.4 + 0.5)

    def test_valid_phrases(self):
        hypothesis = _Hypothesis()
        hypothesis.untranslated_spans = lambda _: [(0, 2), (3, 6)]
        all_phrases_from = [[1, 4], [2], [], [5], [5, 6, 7], [], [7]]
        phrase_spans = StackDecoder.valid_phrases(all_phrases_from, hypothesis)
        self.assertEqual(phrase_spans, [(0, 1), (1, 2), (3, 5), (4, 5), (4, 6)])

    @staticmethod
    def create_fake_phrase_table():
        phrase_table = PhraseTable()
        phrase_table.add(('hovercraft',), ('',), 0.8)
        phrase_table.add(('my', 'hovercraft'), ('', ''), 0.7)
        phrase_table.add(('my', 'cheese'), ('', ''), 0.7)
        phrase_table.add(('is',), ('',), 0.8)
        phrase_table.add(('is',), ('',), 0.5)
        phrase_table.add(('full', 'of'), ('', ''), 0.01)
        phrase_table.add(('full', 'of', 'eels'), ('', '', ''), 0.5)
        phrase_table.add(('full', 'of', 'spam'), ('', ''), 0.5)
        phrase_table.add(('eels',), ('',), 0.5)
        phrase_table.add(('spam',), ('',), 0.5)
        return phrase_table

    @staticmethod
    def create_fake_language_model():
        language_prob = defaultdict(lambda: -999.0)
        language_prob['my',] = log(0.1)
        language_prob['hovercraft',] = log(0.1)
        language_prob['is',] = log(0.1)
        language_prob['full',] = log(0.1)
        language_prob['of',] = log(0.1)
        language_prob['eels',] = log(0.1)
        language_prob['my', 'hovercraft'] = log(0.3)
        language_model = type('', (object,), {'probability': lambda _, phrase: language_prob[phrase]})()
        return language_model