from typing import (
from langchain_core.callbacks import (
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from langchain_community.llms.openai import (
from langchain_community.utils.openai import is_openai_v1
Call out to OpenAI's endpoint async with k unique prompts.