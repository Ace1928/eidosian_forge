from typing import Any, Callable, Iterator, Mapping, Optional
from langchain_core.documents import Document
from langchain_core.utils.utils import guard_import
from langchain_community.document_loaders.base import BaseLoader
class CDKIntegration(BaseEmbeddedIntegration):
    """A wrapper around the CDK integration."""

    def _handle_record(self, record: AirbyteRecordMessage, id: Optional[str]) -> Document:
        if record_handler:
            return record_handler(record, id)
        return Document(page_content='', metadata=record.data)