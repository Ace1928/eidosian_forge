import logging
import threading
import urllib
import warnings
from contextlib import contextmanager
from dataclasses import fields
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import requests
from ray.dashboard.modules.dashboard_sdk import SubmissionClient
from ray.dashboard.utils import (
from ray.util.annotations import DeveloperAPI
from ray.util.state.common import (
from ray.util.state.exception import RayStateApiException, ServerUnavailable
@classmethod
def _make_param(cls, options: Union[ListApiOptions, GetApiOptions]) -> Dict:
    options_dict = {}
    for field in fields(options):
        if field.name == 'filters':
            options_dict['filter_keys'] = []
            options_dict['filter_predicates'] = []
            options_dict['filter_values'] = []
            for filter in options.filters:
                if len(filter) != 3:
                    raise ValueError(f'The given filter has incorrect input type, {filter}. Provide (key, predicate, value) tuples.')
                filter_k, filter_predicate, filter_val = filter
                options_dict['filter_keys'].append(filter_k)
                options_dict['filter_predicates'].append(filter_predicate)
                options_dict['filter_values'].append(filter_val)
            continue
        option_val = getattr(options, field.name)
        if option_val is not None:
            options_dict[field.name] = option_val
    return options_dict