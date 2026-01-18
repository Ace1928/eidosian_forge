import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import Generation, LLMResult
import langchain_community
from langchain_community.callbacks.utils import (
def _summarize_metrics_for_generated_outputs(metrics: Sequence) -> dict:
    pd = import_pandas()
    metrics_df = pd.DataFrame(metrics)
    metrics_summary = metrics_df.describe()
    return metrics_summary.to_dict()