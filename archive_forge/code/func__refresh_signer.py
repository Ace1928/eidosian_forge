import logging
from typing import Any, Dict, List, Optional
import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.utils import get_from_dict_or_env
def _refresh_signer(self) -> None:
    if self.auth.get('signer', None) and hasattr(self.auth['signer'], 'refresh_security_token'):
        self.auth['signer'].refresh_security_token()