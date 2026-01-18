import inspect
import re
import string
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import torch
def _validate_tag(tag: str):
    parts = tag.split('.')
    t = _TAGS
    for part in parts:
        assert set(part) <= set(string.ascii_lowercase + '-'), f'Tag contains invalid characters: {part}'
        if part in t:
            t = t[part]
        else:
            raise ValueError(f'Tag {tag} is not found in registered tags.')