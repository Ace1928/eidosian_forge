import base64
import cloudpickle
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import numpy as np
from triad import assert_or_throw, to_uuid
from triad.utils.convert import get_full_type_path
from tune._utils import product
from tune._utils.math import (
class TransitionChoice(Choice):
    """An ordered random choice of values.
    Please read |SpaceTutorial|.

    :param args: values to choose from
    """

    @property
    def jsondict(self) -> Dict[str, Any]:
        return dict(_expr_='tchoice', values=self.values)

    def __repr__(self) -> str:
        return 'TransitionChoice(' + repr(self._values)[1:-1] + ')'