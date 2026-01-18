import logging
from typing import Any, List
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def _diffbot_api_url(self, diffbot_api: str) -> str:
    return f'https://api.diffbot.com/v3/{diffbot_api}'