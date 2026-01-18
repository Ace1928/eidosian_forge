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
class StringRunMapper(Serializable):
    """Extract items to evaluate from the run object."""

    @property
    def output_keys(self) -> List[str]:
        """The keys to extract from the run."""
        return ['prediction', 'input']

    @abstractmethod
    def map(self, run: Run) -> Dict[str, str]:
        """Maps the Run to a dictionary."""

    def __call__(self, run: Run) -> Dict[str, str]:
        """Maps the Run to a dictionary."""
        if not run.outputs:
            raise ValueError(f'Run {run.id} has no outputs to evaluate.')
        return self.map(run)