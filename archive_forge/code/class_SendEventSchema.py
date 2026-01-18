from datetime import datetime as dt
from typing import List, Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.office365.base import O365BaseTool
from langchain_community.tools.office365.utils import UTC_FORMAT
class SendEventSchema(BaseModel):
    """Input for CreateEvent Tool."""
    body: str = Field(..., description='The message body to include in the event.')
    attendees: List[str] = Field(..., description='The list of attendees for the event.')
    subject: str = Field(..., description='The subject of the event.')
    start_datetime: str = Field(description=' The start datetime for the event in the following format:  YYYY-MM-DDTHH:MM:SS±hh:mm, where "T" separates the date and time  components, and the time zone offset is specified as ±hh:mm.  For example: "2023-06-09T10:30:00+03:00" represents June 9th,  2023, at 10:30 AM in a time zone with a positive offset of 3  hours from Coordinated Universal Time (UTC).')
    end_datetime: str = Field(description=' The end datetime for the event in the following format:  YYYY-MM-DDTHH:MM:SS±hh:mm, where "T" separates the date and time  components, and the time zone offset is specified as ±hh:mm.  For example: "2023-06-09T10:30:00+03:00" represents June 9th,  2023, at 10:30 AM in a time zone with a positive offset of 3  hours from Coordinated Universal Time (UTC).')