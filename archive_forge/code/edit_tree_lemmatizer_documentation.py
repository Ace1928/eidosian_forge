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

        Look up the edit tree identifier for a form/label pair. If the edit
        tree is unknown and "add_label" is set, the edit tree will be added to
        the labels.
        