from typing import Iterator, List
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers.openai_functions import PydanticOutputFunctionsParser
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chains.llm import LLMChain
from langchain.chains.openai_functions.utils import get_llm_kwargs
Create a citation fuzzy match chain.

    Args:
        llm: Language model to use for the chain.

    Returns:
        Chain (LLMChain) that can be used to answer questions with citations.
    