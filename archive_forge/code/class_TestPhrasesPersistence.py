import logging
import unittest
import numpy as np
from gensim.models.phrases import Phrases, FrozenPhrases, _PhrasesTransformation
from gensim.models.phrases import original_scorer
from gensim.test.utils import common_texts, temporary_file, datapath
class TestPhrasesPersistence(PhrasesData, unittest.TestCase):

    def test_save_load_custom_scorer(self):
        """Test saving and loading a Phrases object with a custom scorer."""
        bigram = Phrases(self.sentences, min_count=1, threshold=0.001, scoring=dumb_scorer)
        with temporary_file('test.pkl') as fpath:
            bigram.save(fpath)
            bigram_loaded = Phrases.load(fpath)
        test_sentences = [['graph', 'minors', 'survey', 'human', 'interface', 'system']]
        seen_scores = list(bigram_loaded.find_phrases(test_sentences).values())
        assert all((score == 1 for score in seen_scores))
        assert len(seen_scores) == 3

    def test_save_load(self):
        """Test saving and loading a Phrases object."""
        bigram = Phrases(self.sentences, min_count=1, threshold=1)
        with temporary_file('test.pkl') as fpath:
            bigram.save(fpath)
            bigram_loaded = Phrases.load(fpath)
        test_sentences = [['graph', 'minors', 'survey', 'human', 'interface', 'system']]
        seen_scores = set((round(score, 3) for score in bigram_loaded.find_phrases(test_sentences).values()))
        assert seen_scores == set([5.167, 3.444])

    def test_save_load_with_connector_words(self):
        """Test saving and loading a Phrases object."""
        connector_words = frozenset({'of'})
        bigram = Phrases(self.sentences, min_count=1, threshold=1, connector_words=connector_words)
        with temporary_file('test.pkl') as fpath:
            bigram.save(fpath)
            bigram_loaded = Phrases.load(fpath)
        assert bigram_loaded.connector_words == connector_words

    def test_save_load_string_scoring(self):
        """Test backwards compatibility with a previous version of Phrases with custom scoring."""
        bigram_loaded = Phrases.load(datapath('phrases-scoring-str.pkl'))
        test_sentences = [['graph', 'minors', 'survey', 'human', 'interface', 'system']]
        seen_scores = set((round(score, 3) for score in bigram_loaded.find_phrases(test_sentences).values()))
        assert seen_scores == set([5.167, 3.444])

    def test_save_load_no_scoring(self):
        """Test backwards compatibility with old versions of Phrases with no scoring parameter."""
        bigram_loaded = Phrases.load(datapath('phrases-no-scoring.pkl'))
        test_sentences = [['graph', 'minors', 'survey', 'human', 'interface', 'system']]
        seen_scores = set((round(score, 3) for score in bigram_loaded.find_phrases(test_sentences).values()))
        assert seen_scores == set([5.167, 3.444])

    def test_save_load_no_common_terms(self):
        """Ensure backwards compatibility with old versions of Phrases, before connector_words."""
        bigram_loaded = Phrases.load(datapath('phrases-no-common-terms.pkl'))
        self.assertEqual(bigram_loaded.connector_words, frozenset())
        phraser = FrozenPhrases(bigram_loaded)
        phraser[['human', 'interface', 'survey']]