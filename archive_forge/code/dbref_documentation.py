from __future__ import annotations
from copy import deepcopy
from typing import Any, Mapping, Optional
from bson._helpers import _getstate_slots, _setstate_slots
from bson.son import SON
Support function for `copy.deepcopy()`.