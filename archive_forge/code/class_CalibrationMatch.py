from copy import copy
from dataclasses import dataclass
from typing import Union, Dict, List, Any, Optional, no_type_check
from pyquil.quilatom import (
from pyquil.quilbase import (
@dataclass
class CalibrationMatch:
    cal: Union[DefCalibration, DefMeasureCalibration]
    settings: Dict[Union[FormalArgument, Parameter], Any]