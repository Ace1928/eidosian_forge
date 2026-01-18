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
class StringRunEvaluatorChain(Chain, RunEvaluator):
    """Evaluate Run and optional examples."""
    run_mapper: StringRunMapper
    "Maps the Run to a dictionary with 'input' and 'prediction' strings."
    example_mapper: Optional[StringExampleMapper] = None
    "Maps the Example (dataset row) to a dictionary\n    with a 'reference' string."
    name: str
    'The name of the evaluation metric.'
    string_evaluator: StringEvaluator
    'The evaluation chain.'

    @property
    def input_keys(self) -> List[str]:
        return ['run', 'example']

    @property
    def output_keys(self) -> List[str]:
        return ['feedback']

    def _prepare_input(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        run: Run = inputs['run']
        example: Optional[Example] = inputs.get('example')
        evaluate_strings_inputs = self.run_mapper(run)
        if not self.string_evaluator.requires_input:
            evaluate_strings_inputs.pop('input', None)
        if example and self.example_mapper and self.string_evaluator.requires_reference:
            evaluate_strings_inputs.update(self.example_mapper(example))
        elif self.string_evaluator.requires_reference:
            raise ValueError(f'Evaluator {self.name} requires an reference example from the dataset, but none was provided for run {run.id}.')
        return evaluate_strings_inputs

    def _prepare_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
        evaluation_result = EvaluationResult(key=self.name, comment=output.get('reasoning'), **output)
        if RUN_KEY in output:
            evaluation_result.evaluator_info[RUN_KEY] = output[RUN_KEY]
        return {'feedback': evaluation_result}

    def _call(self, inputs: Dict[str, str], run_manager: Optional[CallbackManagerForChainRun]=None) -> Dict[str, Any]:
        """Call the evaluation chain."""
        evaluate_strings_inputs = self._prepare_input(inputs)
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        chain_output = self.string_evaluator.evaluate_strings(**evaluate_strings_inputs, callbacks=callbacks, include_run_info=True)
        return self._prepare_output(chain_output)

    async def _acall(self, inputs: Dict[str, str], run_manager: Optional[AsyncCallbackManagerForChainRun]=None) -> Dict[str, Any]:
        """Call the evaluation chain."""
        evaluate_strings_inputs = self._prepare_input(inputs)
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        chain_output = await self.string_evaluator.aevaluate_strings(**evaluate_strings_inputs, callbacks=callbacks, include_run_info=True)
        return self._prepare_output(chain_output)

    def _prepare_evaluator_output(self, output: Dict[str, Any]) -> EvaluationResult:
        feedback: EvaluationResult = output['feedback']
        if RUN_KEY not in feedback.evaluator_info:
            feedback.evaluator_info[RUN_KEY] = output[RUN_KEY]
        return feedback

    def evaluate_run(self, run: Run, example: Optional[Example]=None) -> EvaluationResult:
        """Evaluate an example."""
        try:
            result = self({'run': run, 'example': example}, include_run_info=True)
            return self._prepare_evaluator_output(result)
        except Exception as e:
            return EvaluationResult(key=self.string_evaluator.evaluation_name, comment=f'Error evaluating run {run.id}: {e}')

    async def aevaluate_run(self, run: Run, example: Optional[Example]=None) -> EvaluationResult:
        """Evaluate an example."""
        try:
            result = await self.acall({'run': run, 'example': example}, include_run_info=True)
            return self._prepare_evaluator_output(result)
        except Exception as e:
            return EvaluationResult(key=self.string_evaluator.evaluation_name, comment=f'Error evaluating run {run.id}: {e}')

    @classmethod
    def from_run_and_data_type(cls, evaluator: StringEvaluator, run_type: str, data_type: DataType, input_key: Optional[str]=None, prediction_key: Optional[str]=None, reference_key: Optional[str]=None, tags: Optional[List[str]]=None) -> StringRunEvaluatorChain:
        """
        Create a StringRunEvaluatorChain from an evaluator and the run and dataset types.

        This method provides an easy way to instantiate a StringRunEvaluatorChain, by
        taking an evaluator and information about the type of run and the data.
        The method supports LLM and chain runs.

        Args:
            evaluator (StringEvaluator): The string evaluator to use.
            run_type (str): The type of run being evaluated.
                Supported types are LLM and Chain.
            data_type (DataType): The type of dataset used in the run.
            input_key (str, optional): The key used to map the input from the run.
            prediction_key (str, optional): The key used to map the prediction from the run.
            reference_key (str, optional): The key used to map the reference from the dataset.
            tags (List[str], optional): List of tags to attach to the evaluation chain.

        Returns:
            StringRunEvaluatorChain: The instantiated evaluation chain.

        Raises:
            ValueError: If the run type is not supported, or if the evaluator requires a
                reference from the dataset but the reference key is not provided.

        """
        if run_type == 'llm':
            run_mapper: StringRunMapper = LLMStringRunMapper()
        elif run_type == 'chain':
            run_mapper = ChainStringRunMapper(input_key=input_key, prediction_key=prediction_key)
        else:
            raise ValueError(f"Unsupported run type {run_type}. Expected one of 'llm' or 'chain'.")
        if reference_key is not None or data_type in (DataType.llm, DataType.chat) or evaluator.requires_reference:
            example_mapper = StringExampleMapper(reference_key=reference_key)
        elif evaluator.requires_reference:
            raise ValueError(f'Evaluator {evaluator.evaluation_name} requires a reference example from the dataset. Please specify the reference key from amongst the dataset outputs keys.')
        else:
            example_mapper = None
        return cls(name=evaluator.evaluation_name, run_mapper=run_mapper, example_mapper=example_mapper, string_evaluator=evaluator, tags=tags)