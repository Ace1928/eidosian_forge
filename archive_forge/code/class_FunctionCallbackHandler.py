import json
from typing import Any, Callable, List
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run
from langchain_core.utils.input import get_bolded_text, get_colored_text
class FunctionCallbackHandler(BaseTracer):
    """Tracer that calls a function with a single str parameter."""
    name: str = 'function_callback_handler'

    def __init__(self, function: Callable[[str], None], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.function_callback = function

    def _persist_run(self, run: Run) -> None:
        pass

    def get_parents(self, run: Run) -> List[Run]:
        parents = []
        current_run = run
        while current_run.parent_run_id:
            parent = self.run_map.get(str(current_run.parent_run_id))
            if parent:
                parents.append(parent)
                current_run = parent
            else:
                break
        return parents

    def get_breadcrumbs(self, run: Run) -> str:
        parents = self.get_parents(run)[::-1]
        string = ' > '.join((f'{parent.execution_order}:{parent.run_type}:{parent.name}' if i != len(parents) - 1 else f'{parent.execution_order}:{parent.run_type}:{parent.name}' for i, parent in enumerate(parents + [run])))
        return string

    def _on_chain_start(self, run: Run) -> None:
        crumbs = self.get_breadcrumbs(run)
        run_type = run.run_type.capitalize()
        self.function_callback(f'{get_colored_text('[chain/start]', color='green')} ' + get_bolded_text(f'[{crumbs}] Entering {run_type} run with input:\n') + f'{try_json_stringify(run.inputs, '[inputs]')}')

    def _on_chain_end(self, run: Run) -> None:
        crumbs = self.get_breadcrumbs(run)
        run_type = run.run_type.capitalize()
        self.function_callback(f'{get_colored_text('[chain/end]', color='blue')} ' + get_bolded_text(f'[{crumbs}] [{elapsed(run)}] Exiting {run_type} run with output:\n') + f'{try_json_stringify(run.outputs, '[outputs]')}')

    def _on_chain_error(self, run: Run) -> None:
        crumbs = self.get_breadcrumbs(run)
        run_type = run.run_type.capitalize()
        self.function_callback(f'{get_colored_text('[chain/error]', color='red')} ' + get_bolded_text(f'[{crumbs}] [{elapsed(run)}] {run_type} run errored with error:\n') + f'{try_json_stringify(run.error, '[error]')}')

    def _on_llm_start(self, run: Run) -> None:
        crumbs = self.get_breadcrumbs(run)
        inputs = {'prompts': [p.strip() for p in run.inputs['prompts']]} if 'prompts' in run.inputs else run.inputs
        self.function_callback(f'{get_colored_text('[llm/start]', color='green')} ' + get_bolded_text(f'[{crumbs}] Entering LLM run with input:\n') + f'{try_json_stringify(inputs, '[inputs]')}')

    def _on_llm_end(self, run: Run) -> None:
        crumbs = self.get_breadcrumbs(run)
        self.function_callback(f'{get_colored_text('[llm/end]', color='blue')} ' + get_bolded_text(f'[{crumbs}] [{elapsed(run)}] Exiting LLM run with output:\n') + f'{try_json_stringify(run.outputs, '[response]')}')

    def _on_llm_error(self, run: Run) -> None:
        crumbs = self.get_breadcrumbs(run)
        self.function_callback(f'{get_colored_text('[llm/error]', color='red')} ' + get_bolded_text(f'[{crumbs}] [{elapsed(run)}] LLM run errored with error:\n') + f'{try_json_stringify(run.error, '[error]')}')

    def _on_tool_start(self, run: Run) -> None:
        crumbs = self.get_breadcrumbs(run)
        self.function_callback(f'{get_colored_text('[tool/start]', color='green')} ' + get_bolded_text(f'[{crumbs}] Entering Tool run with input:\n') + f'"{run.inputs['input'].strip()}"')

    def _on_tool_end(self, run: Run) -> None:
        crumbs = self.get_breadcrumbs(run)
        if run.outputs:
            self.function_callback(f'{get_colored_text('[tool/end]', color='blue')} ' + get_bolded_text(f'[{crumbs}] [{elapsed(run)}] Exiting Tool run with output:\n') + f'"{run.outputs['output'].strip()}"')

    def _on_tool_error(self, run: Run) -> None:
        crumbs = self.get_breadcrumbs(run)
        self.function_callback(f'{get_colored_text('[tool/error]', color='red')} ' + get_bolded_text(f'[{crumbs}] [{elapsed(run)}] ') + f'Tool run errored with error:\n{run.error}')