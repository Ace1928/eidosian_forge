from typing import Optional
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_community.utilities.scenexplain import SceneXplainAPIWrapper
class SceneXplainInput(BaseModel):
    """Input for SceneXplain."""
    query: str = Field(..., description='The link to the image to explain')