import re
from pathlib import Path
from typing import Iterator, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def _process_acreom_content(self, content: str) -> str:
    content = re.sub('\\s*-\\s\\[\\s\\]\\s.*|\\s*\\[\\s\\]\\s.*', '', content)
    content = re.sub('#', '', content)
    content = re.sub('\\[\\[.*?\\]\\]', '', content)
    return content