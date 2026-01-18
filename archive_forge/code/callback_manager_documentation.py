from __future__ import annotations
import logging # isort:skip
from collections import defaultdict
from inspect import signature
from typing import (
from ..events import Event, ModelEvent
from ..util.functions import get_param_info
 Trigger callbacks for ``attr`` on this object.

        Args:
            attr (str) :
            old (object) :
            new (object) :

        Returns:
            None

        