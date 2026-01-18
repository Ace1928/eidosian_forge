from __future__ import annotations
from typing import List, Optional
import aiohttp
import requests
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
Allow arbitrary types.