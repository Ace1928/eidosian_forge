import logging
import unittest
import numpy as np
from gensim.models.phrases import Phrases, FrozenPhrases, _PhrasesTransformation
from gensim.models.phrases import original_scorer
from gensim.test.utils import common_texts, temporary_file, datapath
class AnalysisTester(_PhrasesTransformation):

    def __init__(self, scores, threshold):
        super().__init__(connector_words={'a', 'the', 'with', 'of'})
        self.scores = scores
        self.threshold = threshold

    def score_candidate(self, word_a, word_b, in_between):
        phrase = '_'.join([word_a] + in_between + [word_b])
        score = self.scores.get(phrase, -1)
        if score > self.threshold:
            return (phrase, score)
        return (None, None)