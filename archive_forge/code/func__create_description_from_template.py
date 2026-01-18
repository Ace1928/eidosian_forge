import json
from typing import Any, Dict, Optional
from langchain_core.callbacks import (
from langchain_core.language_models import BaseLanguageModel
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.vectorstores import VectorStore
from langchain_community.llms.openai import OpenAI
def _create_description_from_template(values: Dict[str, Any]) -> Dict[str, Any]:
    values['description'] = values['template'].format(name=values['name'])
    return values