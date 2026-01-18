from typing import Dict, Any, Tuple, Callable, List, IO, Optional
from types import ModuleType
import sys
from spacy import Language
from spacy.util import SimpleFrozenList
from .util import dict_to_dot, dot_to_dict, matcher_for_regex_patterns
from .util import setup_default_console_logger, LoggerT
def _log_custom_stats(wandb: ModuleType, info: Optional[Dict[str, Any]], matcher: Callable[[str], bool]):
    if info is not None:
        for k, v in info.items():
            if matcher(k):
                wandb.log({k: v})