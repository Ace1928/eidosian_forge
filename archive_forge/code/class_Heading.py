import os
from datetime import datetime
from typing import Dict, Iterable, Optional, Tuple, Union
from typing import List as LList
from urllib.parse import urlparse, urlunparse
from pydantic import ConfigDict, Field, validator
from pydantic.dataclasses import dataclass
import wandb
from . import expr_parsing, gql, internal
from .internal import (
@dataclass(config=dataclass_config)
class Heading(Block):

    @classmethod
    def from_model(cls, model: internal.Heading):
        text = _internal_children_to_text(model.children)
        blocks = None
        if model.collapsed_children:
            blocks = [_lookup(b) for b in model.collapsed_children]
        if model.level == 1:
            return H1(text=text, collapsed_blocks=blocks)
        if model.level == 2:
            return H2(text=text, collapsed_blocks=blocks)
        if model.level == 3:
            return H3(text=text, collapsed_blocks=blocks)