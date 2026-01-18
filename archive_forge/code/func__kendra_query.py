import re
from abc import ABC, abstractmethod
from typing import (
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import (
from langchain_core.retrievers import BaseRetriever
from typing_extensions import Annotated
def _kendra_query(self, query: str) -> Sequence[ResultItem]:
    kendra_kwargs = {'IndexId': self.index_id, 'QueryText': query.strip()[0:999], 'PageSize': self.top_k}
    if self.attribute_filter is not None:
        kendra_kwargs['AttributeFilter'] = self.attribute_filter
    if self.user_context is not None:
        kendra_kwargs['UserContext'] = self.user_context
    response = self.client.retrieve(**kendra_kwargs)
    r_result = RetrieveResult.parse_obj(response)
    if r_result.ResultItems:
        return r_result.ResultItems
    response = self.client.query(**kendra_kwargs)
    q_result = QueryResult.parse_obj(response)
    return q_result.ResultItems