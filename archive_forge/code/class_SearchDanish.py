from typing import Dict
import snowballstemmer
from sphinx.search import SearchLanguage, parse_stop_word
class SearchDanish(SearchLanguage):
    lang = 'da'
    language_name = 'Danish'
    js_stemmer_rawcode = 'danish-stemmer.js'
    stopwords = danish_stopwords

    def init(self, options: Dict) -> None:
        self.stemmer = snowballstemmer.stemmer('danish')

    def stem(self, word: str) -> str:
        return self.stemmer.stemWord(word.lower())