from typing import Dict
import snowballstemmer
from sphinx.search import SearchLanguage, parse_stop_word
class SearchPortuguese(SearchLanguage):
    lang = 'pt'
    language_name = 'Portuguese'
    js_stemmer_rawcode = 'portuguese-stemmer.js'
    stopwords = portuguese_stopwords

    def init(self, options: Dict) -> None:
        self.stemmer = snowballstemmer.stemmer('portuguese')

    def stem(self, word: str) -> str:
        return self.stemmer.stemWord(word.lower())