import json
import logging
from datetime import datetime
from enum import Enum, unique
from typing import Dict, List, Optional, Tuple
import click
import yaml
import ray._private.services as services
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray.util.state import (
from ray.util.state.common import (
from ray.util.state.exception import RayStateApiException
from ray.util.annotations import PublicAPI
def _parse_filter(filter: str) -> Tuple[str, PredicateType, SupportedFilterType]:
    """Parse the filter string to a tuple of key, preciate, and value."""
    predicate = None
    predicate_index = None
    for i in range(len(filter)):
        char = filter[i]
        if char == '=':
            predicate = '='
            predicate_index = (i, i + 1)
            break
        elif char == '!':
            if len(filter) <= i + 1:
                continue
            next_char = filter[i + 1]
            if next_char == '=':
                predicate = '!='
                predicate_index = (i, i + 2)
                break
    if not predicate or not predicate_index:
        raise ValueError(f'The format of a given filter {filter} is invalid: Cannot find the predicate. Please provide key=val or key!=val format string.')
    key, predicate, value = (filter[:predicate_index[0]], filter[predicate_index[0]:predicate_index[1]], filter[predicate_index[1]:])
    assert predicate == '=' or predicate == '!='
    if len(key) == 0 or len(value) == 0:
        raise ValueError(f'The format of a given filter {filter} is invalid: Cannot identify key {key} or value, {value}. Please provide key=val or key!=val format string.')
    return (key, predicate, value)