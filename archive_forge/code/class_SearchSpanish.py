from typing import Dict
import snowballstemmer
from sphinx.search import SearchLanguage, parse_stop_word
class SearchSpanish(SearchLanguage):
    lang = 'es'
    language_name = 'Spanish'
    js_stemmer_rawcode = 'spanish-stemmer.js'
    stopwords = spanish_stopwords

    def init(self, options: Dict) -> None:
        self.stemmer = snowballstemmer.stemmer('spanish')

    def stem(self, word: str) -> str:
        return self.stemmer.stemWord(word.lower())