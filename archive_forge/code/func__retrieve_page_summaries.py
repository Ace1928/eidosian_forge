from typing import Any, Dict, List, Optional
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def _retrieve_page_summaries(self, query_dict: Dict[str, Any]={'page_size': 100}) -> List[Dict[str, Any]]:
    """
        Get all the pages from a Notion database
        OR filter based on specified criteria.
        """
    pages: List[Dict[str, Any]] = []
    while True:
        data = self._request(DATABASE_URL.format(database_id=self.database_id), method='POST', query_dict=query_dict, filter_object=self.filter_object)
        pages.extend(data.get('results'))
        if not data.get('has_more'):
            break
        query_dict['start_cursor'] = data.get('next_cursor')
    return pages