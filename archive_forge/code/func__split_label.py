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
def _split_label(self, label: str) -> Tuple[str, Optional[str]]:
    """Split Entity label into ent_label and ent_id if it contains self.ent_id_sep

        label (str): The value of label in a pattern entry
        RETURNS (tuple): ent_label, ent_id
        """
    if self.ent_id_sep in label:
        ent_label, ent_id = label.rsplit(self.ent_id_sep, 1)
    else:
        ent_label = label
        ent_id = None
    return (ent_label, ent_id)