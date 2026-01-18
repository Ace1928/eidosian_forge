import tempfile
import warnings
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional
import srsly
from ... import util
from ...errors import Errors, Warnings
from ...language import BaseDefaults, Language
from ...scorer import Scorer
from ...tokens import Doc
from ...training import Example, validate_examples
from ...util import DummyTokenizer, load_config_from_str, registry
from ...vocab import Vocab
from .lex_attrs import LEX_ATTRS
from .stop_words import STOP_WORDS
def _get_pkuseg_trie_data(node, path=''):
    data = []
    for c, child_node in sorted(node.children.items()):
        data.extend(_get_pkuseg_trie_data(child_node, path + c))
    if node.isword:
        data.append((path, node.usertag))
    return data