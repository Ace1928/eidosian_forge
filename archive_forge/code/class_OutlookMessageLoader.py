import os
from pathlib import Path
from typing import Any, Iterator, List, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.unstructured import (
class OutlookMessageLoader(BaseLoader):
    """
    Loads Outlook Message files using extract_msg.

    https://github.com/TeamMsgExtractor/msg-extractor
    """

    def __init__(self, file_path: Union[str, Path]):
        """Initialize with a file path.

        Args:
            file_path: The path to the Outlook Message file.
        """
        self.file_path = str(file_path)
        if not os.path.isfile(self.file_path):
            raise ValueError(f'File path {self.file_path} is not a valid file')
        try:
            import extract_msg
        except ImportError:
            raise ImportError('extract_msg is not installed. Please install it with `pip install extract_msg`')

    def lazy_load(self) -> Iterator[Document]:
        import extract_msg
        msg = extract_msg.Message(self.file_path)
        yield Document(page_content=msg.body, metadata={'source': self.file_path, 'subject': msg.subject, 'sender': msg.sender, 'date': msg.date})