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
class BlockQuote(Block):
    text: TextLikeField = ''

    def to_model(self):
        return internal.BlockQuote(children=_text_to_internal_children(self.text))

    @classmethod
    def from_model(cls, model: internal.BlockQuote):
        return cls(text=_internal_children_to_text(model.children))