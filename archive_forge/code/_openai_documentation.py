from __future__ import annotations
import functools
import logging
from collections import defaultdict
from typing import (
from langsmith import run_helpers
Patch the OpenAI client to make it traceable.

    Args:
        client (Union[OpenAI, AsyncOpenAI]): The client to patch.

    Returns:
        Union[OpenAI, AsyncOpenAI]: The patched client.

    