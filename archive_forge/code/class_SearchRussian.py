from typing import Dict
import snowballstemmer
from sphinx.search import SearchLanguage, parse_stop_word
class SearchRussian(SearchLanguage):
    lang = 'ru'
    language_name = 'Russian'
    js_stemmer_rawcode = 'russian-stemmer.js'
    stopwords = russian_stopwords

    def init(self, options: Dict) -> None:
        self.stemmer = snowballstemmer.stemmer('russian')

    def stem(self, word: str) -> str:
        return self.stemmer.stemWord(word.lower())