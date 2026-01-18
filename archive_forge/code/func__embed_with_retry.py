from __future__ import annotations
import logging
import os
import warnings
from typing import (
import numpy as np
from langchain_core._api.deprecation import deprecated
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.utils import get_from_dict_or_env, get_pydantic_field_names
from tenacity import (
from langchain_community.utils.openai import is_openai_v1
@retry_decorator
def _embed_with_retry(**kwargs: Any) -> Any:
    response = embeddings.client.create(**kwargs)
    return _check_response(response, skip_empty=embeddings.skip_empty)