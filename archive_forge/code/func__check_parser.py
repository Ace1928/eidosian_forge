import asyncio
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterator, List, Optional, Union, cast
import aiohttp
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
@staticmethod
def _check_parser(parser: str) -> None:
    """Check that parser is valid for bs4."""
    valid_parsers = ['html.parser', 'lxml', 'xml', 'lxml-xml', 'html5lib']
    if parser not in valid_parsers:
        raise ValueError('`parser` must be one of ' + ', '.join(valid_parsers) + '.')