from typing import Dict
import snowballstemmer
from sphinx.search import SearchLanguage, parse_stop_word
class SearchNorwegian(SearchLanguage):
    lang = 'no'
    language_name = 'Norwegian'
    js_stemmer_rawcode = 'norwegian-stemmer.js'
    stopwords = norwegian_stopwords

    def init(self, options: Dict) -> None:
        self.stemmer = snowballstemmer.stemmer('norwegian')

    def stem(self, word: str) -> str:
        return self.stemmer.stemWord(word.lower())