from itertools import chain, product
from typing import Callable, Iterable, List, Tuple
from nltk.corpus import WordNetCorpusReader, wordnet
from nltk.stem.api import StemmerI
from nltk.stem.porter import PorterStemmer
def _enum_stem_match(enum_hypothesis_list: List[Tuple[int, str]], enum_reference_list: List[Tuple[int, str]], stemmer: StemmerI=PorterStemmer()) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    """
    Stems each word and matches them in hypothesis and reference
    and returns a word mapping between enum_hypothesis_list and
    enum_reference_list based on the enumerated word id. The function also
    returns a enumerated list of unmatched words for hypothesis and reference.

    :param enum_hypothesis_list: enumerated hypothesis list
    :param enum_reference_list: enumerated reference list
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :return: enumerated matched tuples, enumerated unmatched hypothesis tuples,
             enumerated unmatched reference tuples
    """
    stemmed_enum_hypothesis_list = [(word_pair[0], stemmer.stem(word_pair[1])) for word_pair in enum_hypothesis_list]
    stemmed_enum_reference_list = [(word_pair[0], stemmer.stem(word_pair[1])) for word_pair in enum_reference_list]
    return _match_enums(stemmed_enum_hypothesis_list, stemmed_enum_reference_list)