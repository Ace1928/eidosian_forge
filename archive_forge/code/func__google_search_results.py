from typing import Any, Dict, List, Optional
from langchain_core._api.deprecation import deprecated
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
def _google_search_results(self, search_term: str, **kwargs: Any) -> List[dict]:
    cse = self.search_engine.cse()
    if self.siterestrict:
        cse = cse.siterestrict()
    res = cse.list(q=search_term, cx=self.google_cse_id, **kwargs).execute()
    return res.get('items', [])