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
class LLMStringRunMapper(StringRunMapper):
    """Extract items to evaluate from the run object."""

    def serialize_chat_messages(self, messages: List[Dict]) -> str:
        """Extract the input messages from the run."""
        if isinstance(messages, list) and messages:
            if isinstance(messages[0], dict):
                chat_messages = _get_messages_from_run_dict(messages)
            elif isinstance(messages[0], list):
                chat_messages = _get_messages_from_run_dict(messages[0])
            else:
                raise ValueError(f'Could not extract messages to evaluate {messages}')
            return get_buffer_string(chat_messages)
        raise ValueError(f'Could not extract messages to evaluate {messages}')

    def serialize_inputs(self, inputs: Dict) -> str:
        if 'prompts' in inputs:
            input_ = '\n\n'.join(inputs['prompts'])
        elif 'prompt' in inputs:
            input_ = inputs['prompt']
        elif 'messages' in inputs:
            input_ = self.serialize_chat_messages(inputs['messages'])
        else:
            raise ValueError('LLM Run must have either messages or prompts as inputs.')
        return input_

    def serialize_outputs(self, outputs: Dict) -> str:
        if not outputs.get('generations'):
            raise ValueError('Cannot evaluate LLM Run without generations.')
        generations: List[Dict] = outputs['generations']
        if not generations:
            raise ValueError('Cannot evaluate LLM run with empty generations.')
        first_generation: Dict = generations[0]
        if isinstance(first_generation, list):
            first_generation = first_generation[0]
        if 'message' in first_generation:
            output_ = self.serialize_chat_messages([first_generation['message']])
        else:
            output_ = first_generation['text']
        return output_

    def map(self, run: Run) -> Dict[str, str]:
        """Maps the Run to a dictionary."""
        if run.run_type != 'llm':
            raise ValueError('LLM RunMapper only supports LLM runs.')
        elif not run.outputs:
            if run.error:
                raise ValueError(f'Cannot evaluate errored LLM run {run.id}: {run.error}')
            else:
                raise ValueError(f'Run {run.id} has no outputs. Cannot evaluate this run.')
        else:
            try:
                inputs = self.serialize_inputs(run.inputs)
            except Exception as e:
                raise ValueError(f'Could not parse LM input from run inputs {run.inputs}') from e
            try:
                output_ = self.serialize_outputs(run.outputs)
            except Exception as e:
                raise ValueError(f'Could not parse LM prediction from run outputs {run.outputs}') from e
            return {'input': inputs, 'prediction': output_}