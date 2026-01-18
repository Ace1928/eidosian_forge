import logging
import os
import random
import string
import tempfile
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.outputs import LLMResult
from langchain_core.utils import get_from_dict_or_env
from langchain_community.callbacks.utils import (
class MlflowLogger:
    """Callback Handler that logs metrics and artifacts to mlflow server.

    Parameters:
        name (str): Name of the run.
        experiment (str): Name of the experiment.
        tags (dict): Tags to be attached for the run.
        tracking_uri (str): MLflow tracking server uri.

    This handler implements the helper functions to initialize,
    log metrics and artifacts to the mlflow server.
    """

    def __init__(self, **kwargs: Any):
        self.mlflow = import_mlflow()
        if 'DATABRICKS_RUNTIME_VERSION' in os.environ:
            self.mlflow.set_tracking_uri('databricks')
            self.mlf_expid = self.mlflow.tracking.fluent._get_experiment_id()
            self.mlf_exp = self.mlflow.get_experiment(self.mlf_expid)
        else:
            tracking_uri = get_from_dict_or_env(kwargs, 'tracking_uri', 'MLFLOW_TRACKING_URI', '')
            self.mlflow.set_tracking_uri(tracking_uri)
            if (run_id := kwargs.get('run_id')):
                self.mlf_expid = self.mlflow.get_run(run_id).info.experiment_id
            else:
                experiment_name = get_from_dict_or_env(kwargs, 'experiment_name', 'MLFLOW_EXPERIMENT_NAME')
                self.mlf_exp = self.mlflow.get_experiment_by_name(experiment_name)
                if self.mlf_exp is not None:
                    self.mlf_expid = self.mlf_exp.experiment_id
                else:
                    self.mlf_expid = self.mlflow.create_experiment(experiment_name)
        self.start_run(kwargs['run_name'], kwargs['run_tags'], kwargs.get('run_id', None))
        self.dir = kwargs.get('artifacts_dir', '')

    def start_run(self, name: str, tags: Dict[str, str], run_id: Optional[str]=None) -> None:
        """
        If run_id is provided, it will reuse the run with the given run_id.
        Otherwise, it starts a new run, auto generates the random suffix for name.
        """
        if run_id is None:
            if name.endswith('-%'):
                rname = ''.join(random.choices(string.ascii_uppercase + string.digits, k=7))
                name = name[:-1] + rname
            run = self.mlflow.MlflowClient().create_run(self.mlf_expid, run_name=name, tags=tags)
            run_id = run.info.run_id
        self.run_id = run_id

    def finish_run(self) -> None:
        """To finish the run."""
        self.mlflow.end_run()

    def metric(self, key: str, value: float) -> None:
        """To log metric to mlflow server."""
        self.mlflow.log_metric(key, value, run_id=self.run_id)

    def metrics(self, data: Union[Dict[str, float], Dict[str, int]], step: Optional[int]=0) -> None:
        """To log all metrics in the input dict."""
        self.mlflow.log_metrics(data, run_id=self.run_id)

    def jsonf(self, data: Dict[str, Any], filename: str) -> None:
        """To log the input data as json file artifact."""
        self.mlflow.log_dict(data, os.path.join(self.dir, f'{filename}.json'), run_id=self.run_id)

    def table(self, name: str, dataframe: Any) -> None:
        """To log the input pandas dataframe as a html table"""
        self.html(dataframe.to_html(), f'table_{name}')

    def html(self, html: str, filename: str) -> None:
        """To log the input html string as html file artifact."""
        self.mlflow.log_text(html, os.path.join(self.dir, f'{filename}.html'), run_id=self.run_id)

    def text(self, text: str, filename: str) -> None:
        """To log the input text as text file artifact."""
        self.mlflow.log_text(text, os.path.join(self.dir, f'{filename}.txt'), run_id=self.run_id)

    def artifact(self, path: str) -> None:
        """To upload the file from given path as artifact."""
        self.mlflow.log_artifact(path, run_id=self.run_id)

    def langchain_artifact(self, chain: Any) -> None:
        self.mlflow.langchain.log_model(chain, 'langchain-model', run_id=self.run_id)