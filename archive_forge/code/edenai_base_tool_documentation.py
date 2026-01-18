from __future__ import annotations
import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional
import requests
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import root_validator
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
Take a dict response and condense it's data in a human readable string