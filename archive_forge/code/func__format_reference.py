import re
from typing import (
from langchain_core.agents import AgentAction
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseLanguageModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.pydantic_v1 import Extra, Field
from langchain_core.tools import BaseTool
from langchain.callbacks.manager import (
from langchain.chains.llm import LLMChain
from langchain.evaluation.agents.trajectory_eval_prompt import (
from langchain.evaluation.schema import AgentTrajectoryEvaluator, LLMEvalChain
@staticmethod
def _format_reference(reference: Optional[str]) -> str:
    """Format the reference text.

        Args:
            reference (str): The reference text.

        Returns:
            str: The formatted reference text.
        """
    if not reference:
        return ''
    return f'\n\nThe following is the expected answer. Use this to measure correctness:\n[GROUND_TRUTH]\n{reference}\n[END_GROUND_TRUTH]\n'