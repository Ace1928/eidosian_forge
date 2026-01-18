from __future__ import annotations
import os
import uuid
from collections import defaultdict
from datetime import datetime
from time import time
from typing import TYPE_CHECKING, Any, DefaultDict, Dict, List, Optional
import numpy as np
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
Do nothing