from typing import Iterable, Optional
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from requests.exceptions import HTTPError, ReadTimeout
from urllib3.exceptions import ConnectionError
from langchain_community.document_loaders.web_base import WebBaseLoader
@staticmethod
def _format_results(docs: Iterable[Document], query: str) -> str:
    doc_strings = ['\n'.join([doc.metadata['title'], doc.metadata['description']]) for doc in docs if query in doc.metadata['description'] or query in doc.metadata['title']]
    return '\n\n'.join(doc_strings)