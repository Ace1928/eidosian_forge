from itertools import chain, product
from typing import Callable, Iterable, List, Tuple
from nltk.corpus import WordNetCorpusReader, wordnet
from nltk.stem.api import StemmerI
from nltk.stem.porter import PorterStemmer
def _enum_wordnetsyn_match(enum_hypothesis_list: List[Tuple[int, str]], enum_reference_list: List[Tuple[int, str]], wordnet: WordNetCorpusReader=wordnet) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    """
    Matches each word in reference to a word in hypothesis
    if any synonym of a hypothesis word is the exact match
    to the reference word.

    :param enum_hypothesis_list: enumerated hypothesis list
    :param enum_reference_list: enumerated reference list
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    """
    word_match = []
    for i in range(len(enum_hypothesis_list))[::-1]:
        hypothesis_syns = set(chain.from_iterable(((lemma.name() for lemma in synset.lemmas() if lemma.name().find('_') < 0) for synset in wordnet.synsets(enum_hypothesis_list[i][1])))).union({enum_hypothesis_list[i][1]})
        for j in range(len(enum_reference_list))[::-1]:
            if enum_reference_list[j][1] in hypothesis_syns:
                word_match.append((enum_hypothesis_list[i][0], enum_reference_list[j][0]))
                enum_hypothesis_list.pop(i)
                enum_reference_list.pop(j)
                break
    return (word_match, enum_hypothesis_list, enum_reference_list)