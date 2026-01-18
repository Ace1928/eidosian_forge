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
class H3(Heading):
    text: TextLikeField = ''
    collapsed_blocks: Optional[LList['BlockTypes']] = None

    def to_model(self):
        collapsed_children = self.collapsed_blocks
        if collapsed_children is not None:
            collapsed_children = [b.to_model() for b in collapsed_children]
        return internal.Heading(level=3, children=_text_to_internal_children(self.text), collapsed_children=collapsed_children)