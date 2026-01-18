from typing import Any, List, Optional, Sequence
from langchain_core._api import deprecated
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool, Tool
from langchain.agents.agent import Agent, AgentExecutor, AgentOutputParser
from langchain.agents.agent_types import AgentType
from langchain.agents.react.output_parser import ReActOutputParser
from langchain.agents.react.textworld_prompt import TEXTWORLD_PROMPT
from langchain.agents.react.wiki_prompt import WIKI_PROMPT
from langchain.agents.utils import validate_tools_single_input
from langchain.docstore.base import Docstore
@property
def _paragraphs(self) -> List[str]:
    if self.document is None:
        raise ValueError('Cannot get paragraphs without a document')
    return self.document.page_content.split('\n\n')