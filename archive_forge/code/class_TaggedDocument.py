import logging
import os
from collections import namedtuple, defaultdict
from collections.abc import Iterable
from timeit import default_timer
from dataclasses import dataclass
from numpy import zeros, float32 as REAL, vstack, integer, dtype
import numpy as np
from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.utils import deprecated
from gensim.models import Word2Vec, FAST_VERSION  # noqa: F401
from gensim.models.keyedvectors import KeyedVectors, pseudorandom_weak_vector
class TaggedDocument(namedtuple('TaggedDocument', 'words tags')):
    """Represents a document along with a tag, input document format for :class:`~gensim.models.doc2vec.Doc2Vec`.

    A single document, made up of `words` (a list of unicode string tokens) and `tags` (a list of tokens).
    Tags may be one or more unicode string tokens, but typical practice (which will also be the most memory-efficient)
    is for the tags list to include a unique integer id as the only tag.

    Replaces "sentence as a list of words" from :class:`gensim.models.word2vec.Word2Vec`.

    """

    def __str__(self):
        """Human readable representation of the object's state, used for debugging.

        Returns
        -------
        str
           Human readable representation of the object's state (words and tags).

        """
        return '%s<%s, %s>' % (self.__class__.__name__, self.words, self.tags)