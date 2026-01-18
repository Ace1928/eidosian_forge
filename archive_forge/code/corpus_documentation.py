import random
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Iterator, List, Optional, Union
import srsly
from .. import util
from ..errors import Errors, Warnings
from ..tokens import Doc, DocBin
from ..vocab import Vocab
from .augment import dont_augment
from .example import Example
Yield examples from the data.

        nlp (Language): The current nlp object.
        YIELDS (Example): The example objects.

        DOCS: https://spacy.io/api/corpus#plaintextcorpus-call
        