import json
import urllib.request
from typing import Any, List
from langchain_core.documents import Document
from langchain_core.utils import stringify_dict
from langchain_community.document_loaders.base import BaseLoader
Load file