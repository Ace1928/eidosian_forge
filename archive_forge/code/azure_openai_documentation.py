from __future__ import annotations
import os
import warnings
from typing import Callable, Dict, Optional, Union
from langchain_core._api.deprecation import deprecated
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.utils import get_from_dict_or_env
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.utils.openai import is_openai_v1
Validate that api key and python package exists in environment.