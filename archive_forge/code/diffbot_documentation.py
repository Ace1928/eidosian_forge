import logging
from typing import Any, List
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
Extract text from Diffbot on all the URLs and return Documents