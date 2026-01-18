from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union
from langchain_core.pydantic_v1 import root_validator
from langchain_core.callbacks import (
from langchain_core.language_models.llms import LLM
from langchain_community.llms.utils import enforce_stop_tokens
from langchain_core.outputs import GenerationChunk
Yields results objects as they are generated in real time.
        It also calls the callback manager's on_llm_new_token event with
        similar parameters to the OpenAI LLM class method of the same name.
        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.
        Returns:
            A generator representing the stream of tokens being generated.
        Yields:
            A dictionary like object containing a string token.
        Example:
            .. code-block:: python
                from langchain_community.llms import DeepSparse
                llm = DeepSparse(
                    model="zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base_quant-none",
                    streaming=True
                )
                for chunk in llm.stream("Tell me a joke",
                        stop=["'","
"]):
                    print(chunk, end='', flush=True)  # noqa: T201
        