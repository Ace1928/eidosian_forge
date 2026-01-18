from itertools import chain, product
from typing import Callable, Iterable, List, Tuple
from nltk.corpus import WordNetCorpusReader, wordnet
from nltk.stem.api import StemmerI
from nltk.stem.porter import PorterStemmer
def _count_chunks(matches: List[Tuple[int, int]]) -> int:
    """
    Counts the fewest possible number of chunks such that matched unigrams
    of each chunk are adjacent to each other. This is used to calculate the
    fragmentation part of the metric.

    :param matches: list containing a mapping of matched words (output of align_words)
    :return: Number of chunks a sentence is divided into post alignment
    """
    i = 0
    chunks = 1
    while i < len(matches) - 1:
        if matches[i + 1][0] == matches[i][0] + 1 and matches[i + 1][1] == matches[i][1] + 1:
            i += 1
            continue
        i += 1
        chunks += 1
    return chunks