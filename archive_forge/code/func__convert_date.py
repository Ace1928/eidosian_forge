import json
import urllib
from datetime import datetime
from typing import Iterator, List, Optional
from langchain_core.documents import Document
from langchain_core.utils import get_from_env
from langchain_community.document_loaders.base import BaseLoader
def _convert_date(self, date: int) -> str:
    return datetime.fromtimestamp(date / 1000).strftime('%Y-%m-%d %H:%M:%S')