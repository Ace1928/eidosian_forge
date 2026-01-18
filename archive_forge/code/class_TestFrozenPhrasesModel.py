import logging
import unittest
import numpy as np
from gensim.models.phrases import Phrases, FrozenPhrases, _PhrasesTransformation
from gensim.models.phrases import original_scorer
from gensim.test.utils import common_texts, temporary_file, datapath
class TestFrozenPhrasesModel(PhrasesCommon, unittest.TestCase):
    """Test FrozenPhrases models."""

    def setUp(self):
        """Set up FrozenPhrases models for the tests."""
        bigram_phrases = Phrases(self.sentences, min_count=1, threshold=1, connector_words=self.connector_words)
        self.bigram = FrozenPhrases(bigram_phrases)
        bigram_default_phrases = Phrases(self.sentences, connector_words=self.connector_words)
        self.bigram_default = FrozenPhrases(bigram_default_phrases)