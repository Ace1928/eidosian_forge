from __future__ import annotations
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import yaml
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from typing_extensions import TYPE_CHECKING, Literal
from langchain_community.vectorstores.redis.constants import REDIS_VECTOR_DTYPE_MAP
class TextFieldSchema(RedisField):
    """Schema for text fields in Redis."""
    weight: float = 1
    no_stem: bool = False
    phonetic_matcher: Optional[str] = None
    withsuffixtrie: bool = False
    no_index: bool = False
    sortable: Optional[bool] = False

    def as_field(self) -> TextField:
        from redis.commands.search.field import TextField
        return TextField(self.name, weight=self.weight, no_stem=self.no_stem, phonetic_matcher=self.phonetic_matcher, sortable=self.sortable, no_index=self.no_index)