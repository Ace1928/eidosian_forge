import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import Generation, LLMResult
import langchain_community
from langchain_community.callbacks.utils import (
def _get_custom_metrics(self, generation: Generation, prompt_idx: int, gen_idx: int) -> dict:
    """Compute Custom Metrics for an LLM Generated Output

        Args:
            generation (LLMResult): Output generation from an LLM
            prompt_idx (int): List index of the input prompt
            gen_idx (int): List index of the generated output

        Returns:
            dict: A dictionary containing the custom metrics.
        """
    resp = {}
    if self.custom_metrics:
        custom_metrics = self.custom_metrics(generation, prompt_idx, gen_idx)
        resp.update(custom_metrics)
    return resp