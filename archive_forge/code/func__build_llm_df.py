from __future__ import annotations
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_community.callbacks.utils import (
@staticmethod
def _build_llm_df(base_df: pd.DataFrame, base_df_fields: Sequence, rename_map: Mapping) -> pd.DataFrame:
    base_df_fields = [field for field in base_df_fields if field in base_df]
    rename_map = {map_entry_k: map_entry_v for map_entry_k, map_entry_v in rename_map.items() if map_entry_k in base_df_fields}
    llm_df = base_df[base_df_fields].dropna(axis=1)
    if rename_map:
        llm_df = llm_df.rename(rename_map, axis=1)
    return llm_df