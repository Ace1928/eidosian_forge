import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import Generation, LLMResult
import langchain_community
from langchain_community.callbacks.utils import (
def _log_session(self, langchain_asset: Optional[Any]=None) -> None:
    try:
        llm_session_df = self._create_session_analysis_dataframe(langchain_asset)
        self.experiment.log_table('langchain-llm-session.csv', llm_session_df)
    except Exception:
        self.comet_ml.LOGGER.warning('Failed to log session data to Comet', exc_info=True, extra={'show_traceback': True})
    try:
        metadata = {'langchain_version': str(langchain_community.__version__)}
        self.experiment.log_asset_data(self.action_records, 'langchain-action_records.json', metadata=metadata)
    except Exception:
        self.comet_ml.LOGGER.warning('Failed to log session data to Comet', exc_info=True, extra={'show_traceback': True})
    try:
        self._log_visualizations(llm_session_df)
    except Exception:
        self.comet_ml.LOGGER.warning('Failed to log visualizations to Comet', exc_info=True, extra={'show_traceback': True})