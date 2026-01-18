import random
from typing import (
from ...public import PanelMetricsHelper
from .validators import UNDEFINED_TYPE, TypeValidator, Validator
@staticmethod
def _default_panel_layout() -> Dict[str, int]:
    return {'x': 0, 'y': 0, 'w': 8, 'h': 6}