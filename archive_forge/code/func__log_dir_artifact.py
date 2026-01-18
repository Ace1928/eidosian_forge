from typing import Dict, Any, Tuple, Callable, List, IO, Optional
from types import ModuleType
import sys
from spacy import Language
from spacy.util import SimpleFrozenList
from .util import dict_to_dot, dot_to_dict, matcher_for_regex_patterns
from .util import setup_default_console_logger, LoggerT
def _log_dir_artifact(wandb: ModuleType, path: str, name: str, type: str, metadata: Optional[Dict[str, Any]]=None, aliases: Optional[List[str]]=None):
    dataset_artifact = wandb.Artifact(name, type=type, metadata=metadata)
    dataset_artifact.add_dir(path, name=name)
    wandb.log_artifact(dataset_artifact, aliases=aliases)