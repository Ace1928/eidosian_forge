from __future__ import annotations
from abc import abstractmethod
from typing import Any, Dict, List, Optional
from langchain_core.load.dump import dumpd
from langchain_core.load.load import load
from langchain_core.load.serializable import Serializable
from langchain_core.messages import BaseMessage, get_buffer_string, messages_from_dict
from langsmith import EvaluationResult, RunEvaluator
from langsmith.schemas import DataType, Example, Run
from langchain.callbacks.manager import (
from langchain.chains.base import Chain
from langchain.evaluation.schema import StringEvaluator
from langchain.schema import RUN_KEY
class ToolStringRunMapper(StringRunMapper):
    """Map an input to the tool."""

    def map(self, run: Run) -> Dict[str, str]:
        if not run.outputs:
            raise ValueError(f'Run {run.id} has no outputs to evaluate.')
        return {'input': run.inputs['input'], 'prediction': run.outputs['output']}