from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import srsly
from .. import util
from ..errors import Errors
from ..language import Language
from ..matcher import Matcher
from ..scorer import Scorer
from ..symbols import IDS
from ..tokens import Doc, Span
from ..tokens._retokenize import normalize_token_attrs, set_token_attrs
from ..training import Example
from ..util import SimpleFrozenList, registry
from ..vocab import Vocab
from .pipe import Pipe
def _split_morph_attrs(attrs: dict) -> Tuple[dict, dict]:
    """Split entries from a tag map or morph rules dict into to two dicts, one
    with the token-level features (POS, LEMMA) and one with the remaining
    features, which are presumed to be individual MORPH features."""
    other_attrs = {}
    morph_attrs = {}
    for k, v in attrs.items():
        if k in '_' or k in IDS.keys() or k in IDS.values():
            other_attrs[k] = v
        else:
            morph_attrs[k] = v
    return (other_attrs, morph_attrs)