from typing import Iterator, List
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers.openai_functions import PydanticOutputFunctionsParser
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chains.llm import LLMChain
from langchain.chains.openai_functions.utils import get_llm_kwargs
def _get_span(self, quote: str, context: str, errs: int=100) -> Iterator[str]:
    import regex
    minor = quote
    major = context
    errs_ = 0
    s = regex.search(f'({minor}){{e<={errs_}}}', major)
    while s is None and errs_ <= errs:
        errs_ += 1
        s = regex.search(f'({minor}){{e<={errs_}}}', major)
    if s is not None:
        yield from s.spans()