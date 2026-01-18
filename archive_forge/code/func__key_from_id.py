import asyncio
import threading
from collections import defaultdict
from functools import partial
from itertools import groupby
from typing import (
from langchain_core._api.beta_decorator import beta
from langchain_core.runnables.base import (
from langchain_core.runnables.config import RunnableConfig, ensure_config, patch_config
from langchain_core.runnables.utils import ConfigurableFieldSpec, Input, Output
def _key_from_id(id_: str) -> str:
    wout_prefix = id_.split(CONTEXT_CONFIG_PREFIX, maxsplit=1)[1]
    if wout_prefix.endswith(CONTEXT_CONFIG_SUFFIX_GET):
        return wout_prefix[:-len(CONTEXT_CONFIG_SUFFIX_GET)]
    elif wout_prefix.endswith(CONTEXT_CONFIG_SUFFIX_SET):
        return wout_prefix[:-len(CONTEXT_CONFIG_SUFFIX_SET)]
    else:
        raise ValueError(f'Invalid context config id {id_}')