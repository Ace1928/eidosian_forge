import re
from abc import ABC, abstractmethod
from collections import Counter
import functools
import datetime
from typing import Union, List, Optional, Tuple, Set, Any, Dict
import torch
from parlai.core.message import Message
from parlai.utils.misc import warn_once
from parlai.utils.typing import TScalar, TVector
@staticmethod
def _infer_metrics(cli_arg: str) -> Set[str]:
    """
        Parse the CLI metric into a list of metrics we wish to compute.
        """
    col: Set[str] = set()
    names = cli_arg.split(',')
    for n in names:
        if n == 'default':
            col |= DEFAULT_METRICS
        elif n == 'rouge':
            col |= ROUGE_METRICS
        elif n == 'bleu':
            col |= BLEU_METRICS
        elif n == 'all':
            col |= ALL_METRICS
        else:
            col.add(n)
    return col