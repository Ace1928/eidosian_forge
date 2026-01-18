import logging
from datetime import datetime as dt
from typing import Dict, Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.amadeus.base import AmadeusBaseTool
class FlightSearchSchema(BaseModel):
    """Schema for the AmadeusFlightSearch tool."""
    originLocationCode: str = Field(description=" The three letter International Air Transport  Association (IATA) Location Identifier for the  search's origin airport. ")
    destinationLocationCode: str = Field(description=" The three letter International Air Transport  Association (IATA) Location Identifier for the  search's destination airport. ")
    departureDateTimeEarliest: str = Field(description=' The earliest departure datetime from the origin airport  for the flight search in the following format:  "YYYY-MM-DDTHH:MM:SS", where "T" separates the date and time  components. For example: "2023-06-09T10:30:00" represents  June 9th, 2023, at 10:30 AM. ')
    departureDateTimeLatest: str = Field(description=' The latest departure datetime from the origin airport  for the flight search in the following format:  "YYYY-MM-DDTHH:MM:SS", where "T" separates the date and time  components. For example: "2023-06-09T10:30:00" represents  June 9th, 2023, at 10:30 AM. ')
    page_number: int = Field(default=1, description='The specific page number of flight results to retrieve')