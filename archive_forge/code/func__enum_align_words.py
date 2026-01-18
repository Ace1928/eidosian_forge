from itertools import chain, product
from typing import Callable, Iterable, List, Tuple
from nltk.corpus import WordNetCorpusReader, wordnet
from nltk.stem.api import StemmerI
from nltk.stem.porter import PorterStemmer
def _enum_align_words(enum_hypothesis_list: List[Tuple[int, str]], enum_reference_list: List[Tuple[int, str]], stemmer: StemmerI=PorterStemmer(), wordnet: WordNetCorpusReader=wordnet) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    """
    Aligns/matches words in the hypothesis to reference by sequentially
    applying exact match, stemmed match and wordnet based synonym match.
    in case there are multiple matches the match which has the least number
    of crossing is chosen. Takes enumerated list as input instead of
    string input

    :param enum_hypothesis_list: enumerated hypothesis list
    :param enum_reference_list: enumerated reference list
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :return: sorted list of matched tuples, unmatched hypothesis list,
             unmatched reference list
    """
    exact_matches, enum_hypothesis_list, enum_reference_list = _match_enums(enum_hypothesis_list, enum_reference_list)
    stem_matches, enum_hypothesis_list, enum_reference_list = _enum_stem_match(enum_hypothesis_list, enum_reference_list, stemmer=stemmer)
    wns_matches, enum_hypothesis_list, enum_reference_list = _enum_wordnetsyn_match(enum_hypothesis_list, enum_reference_list, wordnet=wordnet)
    return (sorted(exact_matches + stem_matches + wns_matches, key=lambda wordpair: wordpair[0]), enum_hypothesis_list, enum_reference_list)