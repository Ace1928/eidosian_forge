from typing import Dict, Any, Tuple, Callable, List, IO, Optional
from types import ModuleType
import sys
from spacy import Language
from spacy.util import SimpleFrozenList
from .util import dict_to_dot, dot_to_dict, matcher_for_regex_patterns
from .util import setup_default_console_logger, LoggerT
def _log_scores(wandb: ModuleType, info: Optional[Dict[str, Any]]):
    if info is not None:
        score = info['score']
        other_scores = info['other_scores']
        losses = info['losses']
        wandb.log({'score': score})
        if losses:
            wandb.log({f'loss_{k}': v for k, v in losses.items()})
        if isinstance(other_scores, dict):
            wandb.log(other_scores)