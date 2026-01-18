from abc import ABCMeta, abstractmethod
from functools import lru_cache
from itertools import chain
from typing import Dict
from nltk.internals import deprecated, overridden
from nltk.metrics import ConfusionMatrix, accuracy
from nltk.tag.util import untag
@lru_cache(maxsize=1)
def _confusion_cached(self, gold):
    """
        Inner function used after ``gold`` is converted to a
        ``tuple(tuple(tuple(str, str)))``. That way, we can use caching on
        creating a ConfusionMatrix.

        :param gold: The list of tagged sentences to run the tagger with,
            also used as the reference values in the generated confusion matrix.
        :type gold: tuple(tuple(tuple(str, str)))
        :rtype: ConfusionMatrix
        """
    tagged_sents = self.tag_sents((untag(sent) for sent in gold))
    gold_tokens = [token for _word, token in chain.from_iterable(gold)]
    test_tokens = [token for _word, token in chain.from_iterable(tagged_sents)]
    return ConfusionMatrix(gold_tokens, test_tokens)