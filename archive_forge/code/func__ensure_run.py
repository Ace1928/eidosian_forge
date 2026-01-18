from __future__ import annotations
import json
from typing import (
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run
def _ensure_run(self, should_print_url: bool=False) -> None:
    """Ensures an active W&B run exists.

        If not, will start a new run with the provided run_args.
        """
    if self._wandb.run is None:
        run_args: Dict = {**(self._run_args or {})}
        if 'settings' not in run_args:
            run_args['settings'] = {'silent': True}
        self._wandb.init(**run_args)
    if self._wandb.run is not None:
        if should_print_url:
            run_url = self._wandb.run.settings.run_url
            self._wandb.termlog(f'Streaming LangChain activity to W&B at {run_url}\n`WandbTracer` is currently in beta.\nPlease report any issues to https://github.com/wandb/wandb/issues with the tag `langchain`.')
        self._wandb.run._label(repo='langchain')