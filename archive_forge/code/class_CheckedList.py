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
class CheckedList(List):
    items: LList[CheckedListItem] = Field(default_factory=lambda: [CheckedListItem()])

    def to_model(self):
        items = [x.to_model() for x in self.items]
        return internal.List(children=items)