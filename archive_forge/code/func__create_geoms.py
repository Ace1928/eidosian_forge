from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, fields
from functools import cached_property
from typing import TYPE_CHECKING, Literal, cast
from warnings import warn
import numpy as np
import pandas as pd
from .._utils import ensure_xy_location
from .._utils.registry import Registry
from ..exceptions import PlotnineError, PlotnineWarning
from ..iapi import (
from ..mapping.aes import rename_aesthetics
from .guide import guide
def _create_geoms(self, gdefs: Sequence[guide]) -> Sequence[guide]:
    """
        Add geoms to the guide definitions
        """
    return [_g for g in gdefs if (_g := g.create_geoms())]