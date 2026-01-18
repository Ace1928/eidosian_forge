from typing import Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_community.utilities.polygon import PolygonAPIWrapper
class PolygonAggregatesSchema(BaseModel):
    """Input for PolygonAggregates."""
    ticker: str = Field(description='The ticker symbol to fetch aggregates for.')
    timespan: str = Field(description="The size of the time window. Possible values are: second, minute, hour, day, week, month, quarter, year. Default is 'day'")
    timespan_multiplier: int = Field(description="The number of timespans to aggregate. For example, if timespan is 'day' and timespan_multiplier is 1, the result will be daily bars. If timespan is 'day' and timespan_multiplier is 5, the result will be weekly bars.  Default is 1.")
    from_date: str = Field(description='The start of the aggregate time window. Either a date with the format YYYY-MM-DD or a millisecond timestamp.')
    to_date: str = Field(description='The end of the aggregate time window. Either a date with the format YYYY-MM-DD or a millisecond timestamp.')