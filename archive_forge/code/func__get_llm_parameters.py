import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import Generation, LLMResult
import langchain_community
from langchain_community.callbacks.utils import (
def _get_llm_parameters(self, langchain_asset: Any=None) -> dict:
    if not langchain_asset:
        return {}
    try:
        if hasattr(langchain_asset, 'agent'):
            llm_parameters = langchain_asset.agent.llm_chain.llm.dict()
        elif hasattr(langchain_asset, 'llm_chain'):
            llm_parameters = langchain_asset.llm_chain.llm.dict()
        elif hasattr(langchain_asset, 'llm'):
            llm_parameters = langchain_asset.llm.dict()
        else:
            llm_parameters = langchain_asset.dict()
    except Exception:
        return {}
    return llm_parameters