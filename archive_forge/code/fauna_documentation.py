from typing import Iterator, Optional, Sequence
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
Load from `FaunaDB`.

    Attributes:
        query (str): The FQL query string to execute.
        page_content_field (str): The field that contains the content of each page.
        secret (str): The secret key for authenticating to FaunaDB.
        metadata_fields (Optional[Sequence[str]]):
            Optional list of field names to include in metadata.
    