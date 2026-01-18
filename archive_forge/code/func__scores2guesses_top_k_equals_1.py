from collections import Counter
from itertools import islice
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, cast
import numpy as np
import srsly
from thinc.api import Config, Model, NumpyOps, SequenceCategoricalCrossentropy
from thinc.types import Floats2d, Ints2d
from .. import util
from ..errors import Errors
from ..language import Language
from ..tokens import Doc
from ..training import Example, validate_examples, validate_get_examples
from ..vocab import Vocab
from ._edit_tree_internals.edit_trees import EditTrees
from ._edit_tree_internals.schemas import validate_edit_tree
from .lemmatizer import lemmatizer_score
from .trainable_pipe import TrainablePipe
def _scores2guesses_top_k_equals_1(self, docs, scores):
    guesses = []
    for doc, doc_scores in zip(docs, scores):
        doc_guesses = doc_scores.argmax(axis=1)
        doc_guesses = self.numpy_ops.asarray(doc_guesses)
        doc_compat_guesses = []
        for i, token in enumerate(doc):
            tree_id = self.cfg['labels'][doc_guesses[i]]
            if self.trees.apply(tree_id, token.text) is not None:
                doc_compat_guesses.append(tree_id)
            else:
                doc_compat_guesses.append(-1)
        guesses.append(np.array(doc_compat_guesses))
    return guesses