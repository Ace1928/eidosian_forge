import json
import re
from typing import Any
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain.chains.api.openapi.prompts import RESPONSE_TEMPLATE
from langchain.chains.llm import LLMChain
class APIResponderOutputParser(BaseOutputParser):
    """Parse the response and error tags."""

    def _load_json_block(self, serialized_block: str) -> str:
        try:
            response_content = json.loads(serialized_block, strict=False)
            return response_content.get('response', 'ERROR parsing response.')
        except json.JSONDecodeError:
            return 'ERROR parsing response.'
        except:
            raise

    def parse(self, llm_output: str) -> str:
        """Parse the response and error tags."""
        json_match = re.search('```json(.*?)```', llm_output, re.DOTALL)
        if json_match:
            return self._load_json_block(json_match.group(1).strip())
        else:
            raise ValueError(f'No response found in output: {llm_output}.')

    @property
    def _type(self) -> str:
        return 'api_responder'