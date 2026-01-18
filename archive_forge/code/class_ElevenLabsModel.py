import tempfile
from enum import Enum
from typing import Any, Dict, Optional, Union
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import root_validator
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
class ElevenLabsModel(str, Enum):
    """Models available for Eleven Labs Text2Speech."""
    MULTI_LINGUAL = 'eleven_multilingual_v1'
    MONO_LINGUAL = 'eleven_monolingual_v1'