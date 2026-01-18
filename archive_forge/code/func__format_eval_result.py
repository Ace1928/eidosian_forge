from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
from .basic import (Booster, _ConfigAliases, _LGBM_BoosterEvalMethodResultType,
def _format_eval_result(value: _EvalResultTuple, show_stdv: bool) -> str:
    """Format metric string."""
    if len(value) == 4:
        return f"{value[0]}'s {value[1]}: {value[2]:g}"
    elif len(value) == 5:
        if show_stdv:
            return f"{value[0]}'s {value[1]}: {value[2]:g} + {value[4]:g}"
        else:
            return f"{value[0]}'s {value[1]}: {value[2]:g}"
    else:
        raise ValueError('Wrong metric value')