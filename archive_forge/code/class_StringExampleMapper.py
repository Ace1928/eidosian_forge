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
class StringExampleMapper(Serializable):
    """Map an example, or row in the dataset, to the inputs of an evaluation."""
    reference_key: Optional[str] = None

    @property
    def output_keys(self) -> List[str]:
        """The keys to extract from the run."""
        return ['reference']

    def serialize_chat_messages(self, messages: List[Dict]) -> str:
        """Extract the input messages from the run."""
        chat_messages = _get_messages_from_run_dict(messages)
        return get_buffer_string(chat_messages)

    def map(self, example: Example) -> Dict[str, str]:
        """Maps the Example, or dataset row to a dictionary."""
        if not example.outputs:
            raise ValueError(f'Example {example.id} has no outputs to use as a reference.')
        if self.reference_key is None:
            if len(example.outputs) > 1:
                raise ValueError(f'Example {example.id} has multiple outputs, so you must specify a reference_key.')
            else:
                output = list(example.outputs.values())[0]
        elif self.reference_key not in example.outputs:
            raise ValueError(f'Example {example.id} does not have reference key {self.reference_key}.')
        else:
            output = example.outputs[self.reference_key]
        return {'reference': self.serialize_chat_messages([output]) if isinstance(output, dict) and output.get('type') and output.get('data') else output}

    def __call__(self, example: Example) -> Dict[str, str]:
        """Maps the Run and Example to a dictionary."""
        if not example.outputs:
            raise ValueError(f'Example {example.id} has no outputs to use as areference label.')
        return self.map(example)