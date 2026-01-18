from collections import OrderedDict
import os
import torch
from torch.serialization import default_restore_location
from typing import Any, Dict, List
from parlai.core.agents import create_agent
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript
def _load_single_fairseq_checkpoint(self, path: str) -> Dict[str, Any]:
    """
        Loads a fairseq model from file.

        :param path:
            path to file

        :return state:
            loaded fairseq state
        """
    with open(path, 'rb') as f:
        try:
            state = torch.load(f, map_location=lambda s, l: default_restore_location(s, 'cpu'))
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Please install fairseq: https://github.com/pytorch/fairseq#requirements-and-installation')
    return state