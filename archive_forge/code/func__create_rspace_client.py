import os
from typing import Any, Dict, Iterator, List, Optional, Union
from langchain_core.documents import Document
from langchain_core.utils import get_from_dict_or_env
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.base import BaseLoader
def _create_rspace_client(self) -> Any:
    """Create a RSpace client."""
    try:
        from rspace_client.eln import eln, field_content
    except ImportError:
        raise ImportError('You must run `pip install rspace_client`')
    try:
        eln = eln.ELNClient(self.url, self.api_key)
        eln.get_status()
    except Exception:
        raise Exception(f'Unable to initialize client - is url {self.url} or api key  correct?')
    return (eln, field_content.FieldContent)