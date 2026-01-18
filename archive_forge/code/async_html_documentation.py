import asyncio
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterator, List, Optional, Union, cast
import aiohttp
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
Load text from the url(s) in web_path.