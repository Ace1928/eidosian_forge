from sys import maxsize
from nltk.util import trigrams
class TextCat:
    _corpus = None
    fingerprints = {}
    _START_CHAR = '<'
    _END_CHAR = '>'
    last_distances = {}

    def __init__(self):
        if not re:
            raise OSError("classify.textcat requires the regex module that supports unicode. Try '$ pip install regex' and see https://pypi.python.org/pypi/regex for further details.")
        from nltk.corpus import crubadan
        self._corpus = crubadan
        for lang in self._corpus.langs():
            self._corpus.lang_freq(lang)

    def remove_punctuation(self, text):
        """Get rid of punctuation except apostrophes"""
        return re.sub("[^\\P{P}\\']+", '', text)

    def profile(self, text):
        """Create FreqDist of trigrams within text"""
        from nltk import FreqDist, word_tokenize
        clean_text = self.remove_punctuation(text)
        tokens = word_tokenize(clean_text)
        fingerprint = FreqDist()
        for t in tokens:
            token_trigram_tuples = trigrams(self._START_CHAR + t + self._END_CHAR)
            token_trigrams = [''.join(tri) for tri in token_trigram_tuples]
            for cur_trigram in token_trigrams:
                if cur_trigram in fingerprint:
                    fingerprint[cur_trigram] += 1
                else:
                    fingerprint[cur_trigram] = 1
        return fingerprint

    def calc_dist(self, lang, trigram, text_profile):
        """Calculate the "out-of-place" measure between the
        text and language profile for a single trigram"""
        lang_fd = self._corpus.lang_freq(lang)
        dist = 0
        if trigram in lang_fd:
            idx_lang_profile = list(lang_fd.keys()).index(trigram)
            idx_text = list(text_profile.keys()).index(trigram)
            dist = abs(idx_lang_profile - idx_text)
        else:
            dist = maxsize
        return dist

    def lang_dists(self, text):
        """Calculate the "out-of-place" measure between
        the text and all languages"""
        distances = {}
        profile = self.profile(text)
        for lang in self._corpus._all_lang_freq.keys():
            lang_dist = 0
            for trigram in profile:
                lang_dist += self.calc_dist(lang, trigram, profile)
            distances[lang] = lang_dist
        return distances

    def guess_language(self, text):
        """Find the language with the min distance
        to the text and return its ISO 639-3 code"""
        self.last_distances = self.lang_dists(text)
        return min(self.last_distances, key=self.last_distances.get)