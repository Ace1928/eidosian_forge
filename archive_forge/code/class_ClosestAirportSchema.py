from typing import Any, Dict, Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_community.chat_models import ChatOpenAI
from langchain_community.tools.amadeus.base import AmadeusBaseTool
class ClosestAirportSchema(BaseModel):
    """Schema for the AmadeusClosestAirport tool."""
    location: str = Field(description=' The location for which you would like to find the nearest airport  along with optional details such as country, state, region, or  province, allowing for easy processing and identification of  the closest airport. Examples of the format are the following:\n Cali, Colombia\n  Lincoln, Nebraska, United States\n New York, United States\n Sydney, New South Wales, Australia\n Rome, Lazio, Italy\n Toronto, Ontario, Canada\n')