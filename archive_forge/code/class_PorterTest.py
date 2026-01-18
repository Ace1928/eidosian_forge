import unittest
from contextlib import closing
from nltk import data
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
class PorterTest(unittest.TestCase):

    def _vocabulary(self):
        with closing(data.find('stemmers/porter_test/porter_vocabulary.txt').open(encoding='utf-8')) as fp:
            return fp.read().splitlines()

    def _test_against_expected_output(self, stemmer_mode, expected_stems):
        stemmer = PorterStemmer(mode=stemmer_mode)
        for word, true_stem in zip(self._vocabulary(), expected_stems):
            our_stem = stemmer.stem(word)
            assert our_stem == true_stem, '{} should stem to {} in {} mode but got {}'.format(word, true_stem, stemmer_mode, our_stem)

    def test_vocabulary_martin_mode(self):
        """Tests all words from the test vocabulary provided by M Porter

        The sample vocabulary and output were sourced from
        https://tartarus.org/martin/PorterStemmer/voc.txt and
        https://tartarus.org/martin/PorterStemmer/output.txt
        and are linked to from the Porter Stemmer algorithm's homepage
        at https://tartarus.org/martin/PorterStemmer/
        """
        with closing(data.find('stemmers/porter_test/porter_martin_output.txt').open(encoding='utf-8')) as fp:
            self._test_against_expected_output(PorterStemmer.MARTIN_EXTENSIONS, fp.read().splitlines())

    def test_vocabulary_nltk_mode(self):
        with closing(data.find('stemmers/porter_test/porter_nltk_output.txt').open(encoding='utf-8')) as fp:
            self._test_against_expected_output(PorterStemmer.NLTK_EXTENSIONS, fp.read().splitlines())

    def test_vocabulary_original_mode(self):
        with closing(data.find('stemmers/porter_test/porter_original_output.txt').open(encoding='utf-8')) as fp:
            self._test_against_expected_output(PorterStemmer.ORIGINAL_ALGORITHM, fp.read().splitlines())
        self._test_against_expected_output(PorterStemmer.ORIGINAL_ALGORITHM, data.find('stemmers/porter_test/porter_original_output.txt').open(encoding='utf-8').read().splitlines())

    def test_oed_bug(self):
        """Test for bug https://github.com/nltk/nltk/issues/1581

        Ensures that 'oed' can be stemmed without throwing an error.
        """
        assert PorterStemmer().stem('oed') == 'o'

    def test_lowercase_option(self):
        """Test for improvement on https://github.com/nltk/nltk/issues/2507

        Ensures that stems are lowercased when `to_lowercase=True`
        """
        porter = PorterStemmer()
        assert porter.stem('On') == 'on'
        assert porter.stem('I') == 'i'
        assert porter.stem('I', to_lowercase=False) == 'I'
        assert porter.stem('Github') == 'github'
        assert porter.stem('Github', to_lowercase=False) == 'Github'