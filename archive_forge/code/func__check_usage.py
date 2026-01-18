from typing import Dict, Union, Optional, TYPE_CHECKING
from ray.util.annotations import PublicAPI
def _check_usage(self):
    if not (self.hard or self.soft):
        raise ValueError('The `hard` and `soft` parameter of NodeLabelSchedulingStrategy cannot both be empty.')