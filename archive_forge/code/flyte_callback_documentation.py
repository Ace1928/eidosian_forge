from __future__ import annotations
import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Tuple
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_community.callbacks.utils import (
Run on agent action.