from typing import Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_community.utilities.polygon import PolygonAPIWrapper
class PolygonAggregates(BaseTool):
    """
    Tool that gets aggregate bars (stock prices) over a
    given date range for a given ticker from Polygon.
    """
    mode: str = 'get_aggregates'
    name: str = 'polygon_aggregates'
    description: str = "A wrapper around Polygon's Aggregates API. This tool is useful for fetching aggregate bars (stock prices) for a ticker. Input should be the ticker, date range, timespan, and timespan multiplier that you want to get the aggregate bars for."
    args_schema: Type[PolygonAggregatesSchema] = PolygonAggregatesSchema
    api_wrapper: PolygonAPIWrapper

    def _run(self, ticker: str, timespan: str, timespan_multiplier: int, from_date: str, to_date: str, run_manager: Optional[CallbackManagerForToolRun]=None) -> str:
        """Use the Polygon API tool."""
        return self.api_wrapper.run(mode=self.mode, ticker=ticker, timespan=timespan, timespan_multiplier=timespan_multiplier, from_date=from_date, to_date=to_date)