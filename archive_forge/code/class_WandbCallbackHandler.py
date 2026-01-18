import json
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_community.callbacks.utils import (
class WandbCallbackHandler(BaseMetadataCallbackHandler, BaseCallbackHandler):
    """Callback Handler that logs to Weights and Biases.

    Parameters:
        job_type (str): The type of job.
        project (str): The project to log to.
        entity (str): The entity to log to.
        tags (list): The tags to log.
        group (str): The group to log to.
        name (str): The name of the run.
        notes (str): The notes to log.
        visualize (bool): Whether to visualize the run.
        complexity_metrics (bool): Whether to log complexity metrics.
        stream_logs (bool): Whether to stream callback actions to W&B

    This handler will utilize the associated callback method called and formats
    the input of each callback function with metadata regarding the state of LLM run,
    and adds the response to the list of records for both the {method}_records and
    action. It then logs the response using the run.log() method to Weights and Biases.
    """

    def __init__(self, job_type: Optional[str]=None, project: Optional[str]='langchain_callback_demo', entity: Optional[str]=None, tags: Optional[Sequence]=None, group: Optional[str]=None, name: Optional[str]=None, notes: Optional[str]=None, visualize: bool=False, complexity_metrics: bool=False, stream_logs: bool=False) -> None:
        """Initialize callback handler."""
        wandb = import_wandb()
        import_pandas()
        import_textstat()
        spacy = import_spacy()
        super().__init__()
        self.job_type = job_type
        self.project = project
        self.entity = entity
        self.tags = tags
        self.group = group
        self.name = name
        self.notes = notes
        self.visualize = visualize
        self.complexity_metrics = complexity_metrics
        self.stream_logs = stream_logs
        self.temp_dir = tempfile.TemporaryDirectory()
        self.run = wandb.init(job_type=self.job_type, project=self.project, entity=self.entity, tags=self.tags, group=self.group, name=self.name, notes=self.notes)
        warning = 'DEPRECATION: The `WandbCallbackHandler` will soon be deprecated in favor of the `WandbTracer`. Please update your code to use the `WandbTracer` instead.'
        wandb.termwarn(warning, repeat=False)
        self.callback_columns: list = []
        self.action_records: list = []
        self.complexity_metrics = complexity_metrics
        self.visualize = visualize
        self.nlp = spacy.load('en_core_web_sm')

    def _init_resp(self) -> Dict:
        return {k: None for k in self.callback_columns}

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Run when LLM starts."""
        self.step += 1
        self.llm_starts += 1
        self.starts += 1
        resp = self._init_resp()
        resp.update({'action': 'on_llm_start'})
        resp.update(flatten_dict(serialized))
        resp.update(self.get_custom_callback_meta())
        for prompt in prompts:
            prompt_resp = deepcopy(resp)
            prompt_resp['prompts'] = prompt
            self.on_llm_start_records.append(prompt_resp)
            self.action_records.append(prompt_resp)
            if self.stream_logs:
                self.run.log(prompt_resp)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run when LLM generates a new token."""
        self.step += 1
        self.llm_streams += 1
        resp = self._init_resp()
        resp.update({'action': 'on_llm_new_token', 'token': token})
        resp.update(self.get_custom_callback_meta())
        self.on_llm_token_records.append(resp)
        self.action_records.append(resp)
        if self.stream_logs:
            self.run.log(resp)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        self.step += 1
        self.llm_ends += 1
        self.ends += 1
        resp = self._init_resp()
        resp.update({'action': 'on_llm_end'})
        resp.update(flatten_dict(response.llm_output or {}))
        resp.update(self.get_custom_callback_meta())
        for generations in response.generations:
            for generation in generations:
                generation_resp = deepcopy(resp)
                generation_resp.update(flatten_dict(generation.dict()))
                generation_resp.update(analyze_text(generation.text, complexity_metrics=self.complexity_metrics, visualize=self.visualize, nlp=self.nlp, output_dir=self.temp_dir.name))
                self.on_llm_end_records.append(generation_resp)
                self.action_records.append(generation_resp)
                if self.stream_logs:
                    self.run.log(generation_resp)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Run when LLM errors."""
        self.step += 1
        self.errors += 1

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain starts running."""
        self.step += 1
        self.chain_starts += 1
        self.starts += 1
        resp = self._init_resp()
        resp.update({'action': 'on_chain_start'})
        resp.update(flatten_dict(serialized))
        resp.update(self.get_custom_callback_meta())
        chain_input = inputs['input']
        if isinstance(chain_input, str):
            input_resp = deepcopy(resp)
            input_resp['input'] = chain_input
            self.on_chain_start_records.append(input_resp)
            self.action_records.append(input_resp)
            if self.stream_logs:
                self.run.log(input_resp)
        elif isinstance(chain_input, list):
            for inp in chain_input:
                input_resp = deepcopy(resp)
                input_resp.update(inp)
                self.on_chain_start_records.append(input_resp)
                self.action_records.append(input_resp)
                if self.stream_logs:
                    self.run.log(input_resp)
        else:
            raise ValueError('Unexpected data format provided!')

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""
        self.step += 1
        self.chain_ends += 1
        self.ends += 1
        resp = self._init_resp()
        resp.update({'action': 'on_chain_end', 'outputs': outputs['output']})
        resp.update(self.get_custom_callback_meta())
        self.on_chain_end_records.append(resp)
        self.action_records.append(resp)
        if self.stream_logs:
            self.run.log(resp)

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        """Run when chain errors."""
        self.step += 1
        self.errors += 1

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        """Run when tool starts running."""
        self.step += 1
        self.tool_starts += 1
        self.starts += 1
        resp = self._init_resp()
        resp.update({'action': 'on_tool_start', 'input_str': input_str})
        resp.update(flatten_dict(serialized))
        resp.update(self.get_custom_callback_meta())
        self.on_tool_start_records.append(resp)
        self.action_records.append(resp)
        if self.stream_logs:
            self.run.log(resp)

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        """Run when tool ends running."""
        output = str(output)
        self.step += 1
        self.tool_ends += 1
        self.ends += 1
        resp = self._init_resp()
        resp.update({'action': 'on_tool_end', 'output': output})
        resp.update(self.get_custom_callback_meta())
        self.on_tool_end_records.append(resp)
        self.action_records.append(resp)
        if self.stream_logs:
            self.run.log(resp)

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        """Run when tool errors."""
        self.step += 1
        self.errors += 1

    def on_text(self, text: str, **kwargs: Any) -> None:
        """
        Run when agent is ending.
        """
        self.step += 1
        self.text_ctr += 1
        resp = self._init_resp()
        resp.update({'action': 'on_text', 'text': text})
        resp.update(self.get_custom_callback_meta())
        self.on_text_records.append(resp)
        self.action_records.append(resp)
        if self.stream_logs:
            self.run.log(resp)

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run when agent ends running."""
        self.step += 1
        self.agent_ends += 1
        self.ends += 1
        resp = self._init_resp()
        resp.update({'action': 'on_agent_finish', 'output': finish.return_values['output'], 'log': finish.log})
        resp.update(self.get_custom_callback_meta())
        self.on_agent_finish_records.append(resp)
        self.action_records.append(resp)
        if self.stream_logs:
            self.run.log(resp)

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        self.step += 1
        self.tool_starts += 1
        self.starts += 1
        resp = self._init_resp()
        resp.update({'action': 'on_agent_action', 'tool': action.tool, 'tool_input': action.tool_input, 'log': action.log})
        resp.update(self.get_custom_callback_meta())
        self.on_agent_action_records.append(resp)
        self.action_records.append(resp)
        if self.stream_logs:
            self.run.log(resp)

    def _create_session_analysis_df(self) -> Any:
        """Create a dataframe with all the information from the session."""
        pd = import_pandas()
        on_llm_start_records_df = pd.DataFrame(self.on_llm_start_records)
        on_llm_end_records_df = pd.DataFrame(self.on_llm_end_records)
        llm_input_prompts_df = on_llm_start_records_df[['step', 'prompts', 'name']].dropna(axis=1).rename({'step': 'prompt_step'}, axis=1)
        complexity_metrics_columns = []
        visualizations_columns = []
        if self.complexity_metrics:
            complexity_metrics_columns = ['flesch_reading_ease', 'flesch_kincaid_grade', 'smog_index', 'coleman_liau_index', 'automated_readability_index', 'dale_chall_readability_score', 'difficult_words', 'linsear_write_formula', 'gunning_fog', 'text_standard', 'fernandez_huerta', 'szigriszt_pazos', 'gutierrez_polini', 'crawford', 'gulpease_index', 'osman']
        if self.visualize:
            visualizations_columns = ['dependency_tree', 'entities']
        llm_outputs_df = on_llm_end_records_df[['step', 'text', 'token_usage_total_tokens', 'token_usage_prompt_tokens', 'token_usage_completion_tokens'] + complexity_metrics_columns + visualizations_columns].dropna(axis=1).rename({'step': 'output_step', 'text': 'output'}, axis=1)
        session_analysis_df = pd.concat([llm_input_prompts_df, llm_outputs_df], axis=1)
        session_analysis_df['chat_html'] = session_analysis_df[['prompts', 'output']].apply(lambda row: construct_html_from_prompt_and_generation(row['prompts'], row['output']), axis=1)
        return session_analysis_df

    def flush_tracker(self, langchain_asset: Any=None, reset: bool=True, finish: bool=False, job_type: Optional[str]=None, project: Optional[str]=None, entity: Optional[str]=None, tags: Optional[Sequence]=None, group: Optional[str]=None, name: Optional[str]=None, notes: Optional[str]=None, visualize: Optional[bool]=None, complexity_metrics: Optional[bool]=None) -> None:
        """Flush the tracker and reset the session.

        Args:
            langchain_asset: The langchain asset to save.
            reset: Whether to reset the session.
            finish: Whether to finish the run.
            job_type: The job type.
            project: The project.
            entity: The entity.
            tags: The tags.
            group: The group.
            name: The name.
            notes: The notes.
            visualize: Whether to visualize.
            complexity_metrics: Whether to compute complexity metrics.

            Returns:
                None
        """
        pd = import_pandas()
        wandb = import_wandb()
        action_records_table = wandb.Table(dataframe=pd.DataFrame(self.action_records))
        session_analysis_table = wandb.Table(dataframe=self._create_session_analysis_df())
        self.run.log({'action_records': action_records_table, 'session_analysis': session_analysis_table})
        if langchain_asset:
            langchain_asset_path = Path(self.temp_dir.name, 'model.json')
            model_artifact = wandb.Artifact(name='model', type='model')
            model_artifact.add(action_records_table, name='action_records')
            model_artifact.add(session_analysis_table, name='session_analysis')
            try:
                langchain_asset.save(langchain_asset_path)
                model_artifact.add_file(str(langchain_asset_path))
                model_artifact.metadata = load_json_to_dict(langchain_asset_path)
            except ValueError:
                langchain_asset.save_agent(langchain_asset_path)
                model_artifact.add_file(str(langchain_asset_path))
                model_artifact.metadata = load_json_to_dict(langchain_asset_path)
            except NotImplementedError as e:
                print('Could not save model.')
                print(repr(e))
                pass
            self.run.log_artifact(model_artifact)
        if finish or reset:
            self.run.finish()
            self.temp_dir.cleanup()
            self.reset_callback_meta()
        if reset:
            self.__init__(job_type=job_type if job_type else self.job_type, project=project if project else self.project, entity=entity if entity else self.entity, tags=tags if tags else self.tags, group=group if group else self.group, name=name if name else self.name, notes=notes if notes else self.notes, visualize=visualize if visualize else self.visualize, complexity_metrics=complexity_metrics if complexity_metrics else self.complexity_metrics)