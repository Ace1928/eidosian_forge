import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import srsly
from ..errors import Errors, Warnings
from ..language import Language
from ..matcher import Matcher, PhraseMatcher
from ..matcher.levenshtein import levenshtein_compare
from ..scorer import get_ner_prf
from ..tokens import Doc, Span
from ..training import Example
from ..util import SimpleFrozenList, ensure_path, from_disk, registry, to_disk
from .pipe import Pipe
def _create_label(self, label: Any, ent_id: Any) -> str:
    """Join Entity label with ent_id if the pattern has an `id` attribute
        If ent_id is not a string, the label is returned as is.

        label (str): The label to set for ent.label_
        ent_id (str): The label
        RETURNS (str): The ent_label joined with configured `ent_id_sep`
        """
    if isinstance(ent_id, str):
        label = f'{label}{self.ent_id_sep}{ent_id}'
    return label