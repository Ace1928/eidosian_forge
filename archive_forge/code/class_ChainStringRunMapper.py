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
class ChainStringRunMapper(StringRunMapper):
    """Extract items to evaluate from the run object from a chain."""
    input_key: Optional[str] = None
    "The key from the model Run's inputs to use as the eval input.\n    If not provided, will use the only input key or raise an\n    error if there are multiple."
    prediction_key: Optional[str] = None
    "The key from the model Run's outputs to use as the eval prediction.\n    If not provided, will use the only output key or raise an error\n    if there are multiple."

    def _get_key(self, source: Dict, key: Optional[str], which: str) -> str:
        if key is not None:
            return source[key]
        elif len(source) == 1:
            return next(iter(source.values()))
        else:
            raise ValueError(f'Could not map run {which} with multiple keys: {source}\nPlease manually specify a {which}_key')

    def map(self, run: Run) -> Dict[str, str]:
        """Maps the Run to a dictionary."""
        if not run.outputs:
            raise ValueError(f'Run with ID {run.id} lacks outputs required for evaluation. Ensure the Run has valid outputs.')
        if self.input_key is not None and self.input_key not in run.inputs:
            raise ValueError(f"Run with ID {run.id} is missing the expected input key '{self.input_key}'.\nAvailable input keys in this Run  are: {run.inputs.keys()}.\nAdjust the evaluator's input_key or ensure your input data includes key '{self.input_key}'.")
        elif self.prediction_key is not None and self.prediction_key not in run.outputs:
            available_keys = ', '.join(run.outputs.keys())
            raise ValueError(f"Run with ID {run.id} doesn't have the expected prediction key '{self.prediction_key}'. Available prediction keys in this Run are: {available_keys}. Adjust the evaluator's prediction_key or ensure the Run object's outputs the expected key.")
        else:
            input_ = self._get_key(run.inputs, self.input_key, 'input')
            prediction = self._get_key(run.outputs, self.prediction_key, 'prediction')
            return {'input': input_, 'prediction': prediction}