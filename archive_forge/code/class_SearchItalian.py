from typing import Dict
import snowballstemmer
from sphinx.search import SearchLanguage, parse_stop_word
class SearchItalian(SearchLanguage):
    lang = 'it'
    language_name = 'Italian'
    js_stemmer_rawcode = 'italian-stemmer.js'
    stopwords = italian_stopwords

    def init(self, options: Dict) -> None:
        self.stemmer = snowballstemmer.stemmer('italian')

    def stem(self, word: str) -> str:
        return self.stemmer.stemWord(word.lower())