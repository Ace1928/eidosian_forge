import json
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Type, Union
from langchain_core.output_parsers import (
from langchain_core.output_parsers.openai_functions import (
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import (
from langchain.output_parsers import (
def _create_openai_functions_structured_output_runnable(output_schema: Union[Dict[str, Any], Type[BaseModel]], llm: Runnable, prompt: Optional[BasePromptTemplate]=None, *, output_parser: Optional[Union[BaseOutputParser, BaseGenerationOutputParser]]=None, **llm_kwargs: Any) -> Runnable:
    if isinstance(output_schema, dict):
        function: Any = {'name': 'output_formatter', 'description': 'Output formatter. Should always be used to format your response to the user.', 'parameters': output_schema}
    else:

        class _OutputFormatter(BaseModel):
            """Output formatter. Should always be used to format your response to the user."""
            output: output_schema
        function = _OutputFormatter
        output_parser = output_parser or PydanticAttrOutputFunctionsParser(pydantic_schema=_OutputFormatter, attr_name='output')
    return create_openai_fn_runnable([function], llm, prompt=prompt, output_parser=output_parser, **llm_kwargs)