import logging
import unittest
from gensim.corpora.dictionary import Dictionary
from gensim.corpora.hashdictionary import HashDictionary
from gensim.topic_coherence import probability_estimation
class TestProbabilityEstimationWithNormalDictionary(BaseTestCases.ProbabilityEstimationBase):

    def setup_dictionary(self):
        self.dictionary = Dictionary(self.texts)
        self.dictionary.id2token = {v: k for k, v in self.dictionary.token2id.items()}