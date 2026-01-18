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
def _scores2guesses_top_k_guardrail(self, docs, scores):
    guesses = []
    for doc, doc_scores in zip(docs, scores):
        doc_guesses = np.argsort(doc_scores)[..., :-self.top_k - 1:-1]
        doc_guesses = self.numpy_ops.asarray(doc_guesses)
        doc_compat_guesses = []
        for token, candidates in zip(doc, doc_guesses):
            tree_id = -1
            for candidate in candidates:
                candidate_tree_id = self.cfg['labels'][candidate]
                if self.trees.apply(candidate_tree_id, token.text) is not None:
                    tree_id = candidate_tree_id
                    break
            doc_compat_guesses.append(tree_id)
        guesses.append(np.array(doc_compat_guesses))
    return guesses