from typing import Any, Dict, List, Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Extra, Field
from langchain_community.tools.office365.base import O365BaseTool
from langchain_community.tools.office365.utils import UTC_FORMAT, clean_body
class SearchEmailsInput(BaseModel):
    """Input for SearchEmails Tool."""
    'From https://learn.microsoft.com/en-us/graph/search-query-parameter'
    folder: str = Field(default=None, description=' If the user wants to search in only one folder, the name of the folder. Default folders are "inbox", "drafts", "sent items", "deleted ttems", but users can search custom folders as well.')
    query: str = Field(description='The Microsoift Graph v1.0 $search query. Example filters include from:sender, from:sender, to:recipient, subject:subject, recipients:list_of_recipients, body:excitement, importance:high, received>2022-12-01, received<2021-12-01, sent>2022-12-01, sent<2021-12-01, hasAttachments:true  attachment:api-catalog.md, cc:samanthab@contoso.com, bcc:samanthab@contoso.com, body:excitement date range example: received:2023-06-08..2023-06-09  matching example: from:amy OR from:david.')
    max_results: int = Field(default=10, description='The maximum number of results to return.')
    truncate: bool = Field(default=True, description='Whether the email body is truncated to meet token number limits. Set to False for searches that will retrieve small messages, otherwise, set to True')