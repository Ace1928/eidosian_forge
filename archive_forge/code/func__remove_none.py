import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from huggingface_hub.utils import logging, yaml_dump
def _remove_none(obj):
    """
    Recursively remove `None` values from a dict. Borrowed from: https://stackoverflow.com/a/20558778
    """
    if isinstance(obj, (list, tuple, set)):
        return type(obj)((_remove_none(x) for x in obj if x is not None))
    elif isinstance(obj, dict):
        return type(obj)(((_remove_none(k), _remove_none(v)) for k, v in obj.items() if k is not None and v is not None))
    else:
        return obj