from typing import Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_community.utilities.reddit_search import RedditSearchAPIWrapper
class RedditSearchSchema(BaseModel):
    """Input for Reddit search."""
    query: str = Field(description="should be query string that post title should         contain, or '*' if anything is allowed.")
    sort: str = Field(description='should be sort method, which is one of: "relevance"         , "hot", "top", "new", or "comments".')
    time_filter: str = Field(description='should be time period to filter by, which is         one of "all", "day", "hour", "month", "week", or "year"')
    subreddit: str = Field(description='should be name of subreddit, like "all" for         r/all')
    limit: str = Field(description='a positive integer indicating the maximum number         of results to return')