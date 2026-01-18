from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
from langchain_core._api import deprecated
from langchain_core.load import Serializable
from langchain_core.messages import (
from langchain_core.messages.base import get_msg_title_repr
from langchain_core.prompt_values import ChatPromptValue, ImageURL, PromptValue
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.prompts.image import ImagePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.string import StringPromptTemplate, get_template_variables
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.utils import get_colored_text
from langchain_core.utils.interactive_env import is_interactive_env
class HumanMessagePromptTemplate(_StringImageMessagePromptTemplate):
    """Human message prompt template. This is a message sent from the user."""
    _msg_class: Type[BaseMessage] = HumanMessage