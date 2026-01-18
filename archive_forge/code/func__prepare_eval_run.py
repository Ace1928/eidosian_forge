from __future__ import annotations
import concurrent.futures
import dataclasses
import functools
import inspect
import logging
import uuid
from datetime import datetime, timezone
from typing import (
from langchain_core._api import warn_deprecated
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, messages_from_dict
from langchain_core.outputs import ChatResult, LLMResult
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.runnables import config as runnable_config
from langchain_core.runnables import utils as runnable_utils
from langchain_core.tracers.evaluation import (
from langchain_core.tracers.langchain import LangChainTracer
from langsmith.client import Client
from langsmith.env import get_git_info, get_langchain_env_var_metadata
from langsmith.evaluation import (
from langsmith.evaluation import (
from langsmith.run_helpers import as_runnable, is_traceable_function
from langsmith.schemas import Dataset, DataType, Example, Run, TracerSession
from langsmith.utils import LangSmithError
from requests import HTTPError
from typing_extensions import TypedDict
from langchain.callbacks.manager import Callbacks
from langchain.chains.base import Chain
from langchain.evaluation.loading import load_evaluator
from langchain.evaluation.schema import (
from langchain.smith import evaluation as smith_eval
from langchain.smith.evaluation import config as smith_eval_config
from langchain.smith.evaluation import name_generation, progress
def _prepare_eval_run(client: Client, dataset_name: str, llm_or_chain_factory: MODEL_OR_CHAIN_FACTORY, project_name: str, project_metadata: Optional[Dict[str, Any]]=None, tags: Optional[List[str]]=None, dataset_version: Optional[Union[str, datetime]]=None) -> Tuple[MCF, TracerSession, Dataset, List[Example]]:
    wrapped_model = _wrap_in_chain_factory(llm_or_chain_factory, dataset_name)
    dataset = client.read_dataset(dataset_name=dataset_name)
    examples = list(client.list_examples(dataset_id=dataset.id, as_of=dataset_version))
    if not examples:
        raise ValueError(f'Dataset {dataset_name} has no example rows.')
    modified_at = [ex.modified_at for ex in examples if ex.modified_at]
    max_modified_at = max(modified_at) if modified_at else None
    inferred_version = max_modified_at.isoformat() if max_modified_at else None
    try:
        project_metadata = project_metadata or {}
        git_info = get_git_info()
        if git_info:
            project_metadata = {**project_metadata, 'git': git_info}
        project_metadata['dataset_version'] = inferred_version
        project = client.create_project(project_name, reference_dataset_id=dataset.id, project_extra={'tags': tags} if tags else {}, metadata=project_metadata)
    except (HTTPError, ValueError, LangSmithError) as e:
        if 'already exists ' not in str(e):
            raise e
        uid = uuid.uuid4()
        example_msg = f'\nrun_on_dataset(\n    ...\n    project_name="{project_name} - {uid}", # Update since {project_name} already exists\n)\n'
        raise ValueError(f'Test project {project_name} already exists. Please use a different name:\n\n{example_msg}')
    comparison_url = dataset.url + f'/compare?selectedSessions={project.id}'
    print(f"View the evaluation results for project '{project_name}' at:\n{comparison_url}\n\nView all tests for Dataset {dataset_name} at:\n{dataset.url}", flush=True)
    return (wrapped_model, project, dataset, examples)