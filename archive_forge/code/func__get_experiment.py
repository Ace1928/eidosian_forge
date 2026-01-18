import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import Generation, LLMResult
import langchain_community
from langchain_community.callbacks.utils import (
def _get_experiment(workspace: Optional[str]=None, project_name: Optional[str]=None) -> Any:
    comet_ml = import_comet_ml()
    experiment = comet_ml.Experiment(workspace=workspace, project_name=project_name)
    return experiment