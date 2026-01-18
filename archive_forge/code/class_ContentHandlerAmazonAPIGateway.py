from typing import Any, Dict, List, Mapping, Optional
import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Extra
from langchain_community.llms.utils import enforce_stop_tokens
class ContentHandlerAmazonAPIGateway:
    """Adapter to prepare the inputs from Langchain to a format
    that LLM model expects.

    It also provides helper function to extract
    the generated text from the model response."""

    @classmethod
    def transform_input(cls, prompt: str, model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return {'inputs': prompt, 'parameters': model_kwargs}

    @classmethod
    def transform_output(cls, response: Any) -> str:
        return response.json()[0]['generated_text']