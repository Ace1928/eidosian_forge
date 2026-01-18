from typing import Dict
import snowballstemmer
from sphinx.search import SearchLanguage, parse_stop_word
class SearchHungarian(SearchLanguage):
    lang = 'hu'
    language_name = 'Hungarian'
    js_stemmer_rawcode = 'hungarian-stemmer.js'
    stopwords = hungarian_stopwords

    def init(self, options: Dict) -> None:
        self.stemmer = snowballstemmer.stemmer('hungarian')

    def stem(self, word: str) -> str:
        return self.stemmer.stemWord(word.lower())