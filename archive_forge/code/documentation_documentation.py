from __future__ import annotations
import inspect
import warnings
from collections import defaultdict
from functools import lru_cache
from typing import Callable

    Generates documentation for any function.
    Parameters:
        fn: Function to document
    Returns:
        description: General description of fn
        parameters: A list of dicts for each parameter, storing data for the parameter name, annotation and doc
        return: A dict storing data for the returned annotation and doc
        example: Code for an example use of the fn
    