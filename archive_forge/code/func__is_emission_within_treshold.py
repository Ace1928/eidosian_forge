import math
import re
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Union
from ..repocard_data import ModelCardData
def _is_emission_within_treshold(model_info: 'ModelInfo', minimum_threshold: float, maximum_threshold: float) -> bool:
    """Checks if a model's emission is within a given threshold.

    Args:
        model_info (`ModelInfo`):
            A model info object containing the model's emission information.
        minimum_threshold (`float`):
            A minimum carbon threshold to filter by, such as 1.
        maximum_threshold (`float`):
            A maximum carbon threshold to filter by, such as 10.

    Returns:
        `bool`: Whether the model's emission is within the given threshold.
    """
    if minimum_threshold is None and maximum_threshold is None:
        raise ValueError('Both `minimum_threshold` and `maximum_threshold` cannot both be `None`')
    if minimum_threshold is None:
        minimum_threshold = -1
    if maximum_threshold is None:
        maximum_threshold = math.inf
    card_data = getattr(model_info, 'card_data', None)
    if card_data is None or not isinstance(card_data, (dict, ModelCardData)):
        return False
    emission = card_data.get('co2_eq_emissions', None)
    if isinstance(emission, dict):
        emission = emission['emissions']
    if not emission:
        return False
    matched = re.search('\\d+\\.\\d+|\\d+', str(emission))
    if matched is None:
        return False
    emission_value = float(matched.group(0))
    return minimum_threshold <= emission_value <= maximum_threshold