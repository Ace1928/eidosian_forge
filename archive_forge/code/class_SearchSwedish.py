from typing import Dict
import snowballstemmer
from sphinx.search import SearchLanguage, parse_stop_word
class SearchSwedish(SearchLanguage):
    lang = 'sv'
    language_name = 'Swedish'
    js_stemmer_rawcode = 'swedish-stemmer.js'
    stopwords = swedish_stopwords

    def init(self, options: Dict) -> None:
        self.stemmer = snowballstemmer.stemmer('swedish')

    def stem(self, word: str) -> str:
        return self.stemmer.stemWord(word.lower())