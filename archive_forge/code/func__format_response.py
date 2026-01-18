import json
from typing import Dict, Iterator, List, Optional
from urllib.parse import quote
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
def _format_response(self, query: str, response: requests.Response) -> str:
    content = json.loads(response.content)
    if not content:
        return f"No Merriam-Webster definition was found for query '{query}'."
    if isinstance(content[0], str):
        result = f"No Merriam-Webster definition was found for query '{query}'.\n"
        if len(content) > 1:
            alternatives = [f'{i + 1}. {content[i]}' for i in range(len(content))]
            result += 'You can try one of the following alternative queries:\n\n'
            result += '\n'.join(alternatives)
        else:
            result += f"Did you mean '{content[0]}'?"
    else:
        result = self._format_definitions(query, content)
    return result