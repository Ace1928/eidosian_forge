from typing import Dict, Union, Optional, TYPE_CHECKING
from ray.util.annotations import PublicAPI
def _validate_label_match_operator_values(values, operator):
    if not values:
        raise ValueError(f'The variadic parameter of the {operator} operator must be a non-empty tuple: e.g. {operator}("value1", "value2").')
    index = 0
    for value in values:
        if not isinstance(value, str):
            raise ValueError(f'Type of value in position {index} for the {operator} operator must be str (e.g. {operator}("value1", "value2")) but got {str(value)} of type {type(value)}.')
        index = index + 1