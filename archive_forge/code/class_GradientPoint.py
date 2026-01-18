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
class GradientPoint(Base):
    color: str
    offset: float = Field(0, ge=0, le=100)

    @validator('color')
    def validate_color(cls, v):
        if not internal.is_valid_color(v):
            raise ValueError('invalid color, value should be hex, rgb, or rgba')
        return v

    def to_model(self):
        return internal.GradientPoint(color=self.color, offset=self.offset)

    @classmethod
    def from_model(cls, model: internal.GradientPoint):
        return cls(color=model.color, offset=model.offset)