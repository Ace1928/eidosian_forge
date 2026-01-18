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
class FlatVectorField(RedisVectorField):
    """Schema for flat vector fields in Redis."""
    algorithm: Literal['FLAT'] = 'FLAT'
    block_size: Optional[int] = None

    def as_field(self) -> VectorField:
        from redis.commands.search.field import VectorField
        field_data = super()._fields()
        if self.block_size is not None:
            field_data['BLOCK_SIZE'] = self.block_size
        return VectorField(self.name, self.algorithm, field_data)