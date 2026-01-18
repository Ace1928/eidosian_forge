import logging
import unittest
from gensim.corpora.dictionary import Dictionary
from gensim.corpora.hashdictionary import HashDictionary
from gensim.topic_coherence import probability_estimation
class TestProbabilityEstimation(BaseTestCases.ProbabilityEstimationBase):

    def setup_dictionary(self):
        self.dictionary = HashDictionary(self.texts)