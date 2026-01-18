from typing import Optional, Type
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_community.utilities.google_places_api import GooglePlacesAPIWrapper
class GooglePlacesSchema(BaseModel):
    """Input for GooglePlacesTool."""
    query: str = Field(..., description='Query for google maps')