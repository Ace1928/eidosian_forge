from typing import Any, Dict, List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def _set_dsn(self) -> None:
    if self.connection_string:
        self.dsn = self.connection_string
    elif self.tns_name:
        self.dsn = self.tns_name