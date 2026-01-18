from typing import Tuple
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
class FinishedOutputParser(BaseOutputParser[Tuple[str, bool]]):
    """Output parser that checks if the output is finished."""
    finished_value: str = 'FINISHED'
    'Value that indicates the output is finished.'

    def parse(self, text: str) -> Tuple[str, bool]:
        cleaned = text.strip()
        finished = self.finished_value in cleaned
        return (cleaned.replace(self.finished_value, ''), finished)