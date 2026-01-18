import io
import json
from abc import abstractmethod
from typing import Any, Dict, Generic, Iterator, List, Mapping, Optional, TypeVar, Union
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Extra, root_validator
from langchain_community.llms.utils import enforce_stop_tokens
class ContentHandlerBase(Generic[INPUT_TYPE, OUTPUT_TYPE]):
    """Handler class to transform input from LLM to a
    format that SageMaker endpoint expects.

    Similarly, the class handles transforming output from the
    SageMaker endpoint to a format that LLM class expects.
    """
    '\n    Example:\n        .. code-block:: python\n\n            class ContentHandler(ContentHandlerBase):\n                content_type = "application/json"\n                accepts = "application/json"\n\n                def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:\n                    input_str = json.dumps({prompt: prompt, **model_kwargs})\n                    return input_str.encode(\'utf-8\')\n                \n                def transform_output(self, output: bytes) -> str:\n                    response_json = json.loads(output.read().decode("utf-8"))\n                    return response_json[0]["generated_text"]\n    '
    content_type: Optional[str] = 'text/plain'
    'The MIME type of the input data passed to endpoint'
    accepts: Optional[str] = 'text/plain'
    'The MIME type of the response data returned from endpoint'

    @abstractmethod
    def transform_input(self, prompt: INPUT_TYPE, model_kwargs: Dict) -> bytes:
        """Transforms the input to a format that model can accept
        as the request Body. Should return bytes or seekable file
        like object in the format specified in the content_type
        request header.
        """

    @abstractmethod
    def transform_output(self, output: bytes) -> OUTPUT_TYPE:
        """Transforms the output from the model to string that
        the LLM class expects.
        """