from datetime import datetime as dt
from typing import Any, Dict, List, Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Extra, Field
from langchain_community.tools.office365.base import O365BaseTool
from langchain_community.tools.office365.utils import UTC_FORMAT, clean_body
class SearchEventsInput(BaseModel):
    """Input for SearchEmails Tool.

    From https://learn.microsoft.com/en-us/graph/search-query-parameter"""
    start_datetime: str = Field(description=' The start datetime for the search query in the following format:  YYYY-MM-DDTHH:MM:SS±hh:mm, where "T" separates the date and time  components, and the time zone offset is specified as ±hh:mm.  For example: "2023-06-09T10:30:00+03:00" represents June 9th,  2023, at 10:30 AM in a time zone with a positive offset of 3  hours from Coordinated Universal Time (UTC).')
    end_datetime: str = Field(description=' The end datetime for the search query in the following format:  YYYY-MM-DDTHH:MM:SS±hh:mm, where "T" separates the date and time  components, and the time zone offset is specified as ±hh:mm.  For example: "2023-06-09T10:30:00+03:00" represents June 9th,  2023, at 10:30 AM in a time zone with a positive offset of 3  hours from Coordinated Universal Time (UTC).')
    max_results: int = Field(default=10, description='The maximum number of results to return.')
    truncate: bool = Field(default=True, description="Whether the event's body is truncated to meet token number limits. Set to False for searches that will retrieve small events, otherwise, set to True.")