from __future__ import annotations
import functools
import logging
from collections import defaultdict
from typing import (
from langsmith import run_helpers
def _strip_not_given(d: dict) -> dict:
    try:
        not_given = _get_not_given()
        if not_given is None:
            return d
        return {k: v for k, v in d.items() if not isinstance(v, not_given)}
    except Exception as e:
        logger.error(f'Error stripping NotGiven: {e}')
        return d