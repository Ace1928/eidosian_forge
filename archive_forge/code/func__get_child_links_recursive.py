from __future__ import annotations
import asyncio
import logging
import re
from typing import (
import requests
from langchain_core.documents import Document
from langchain_core.utils.html import extract_sub_links
from langchain_community.document_loaders.base import BaseLoader
def _get_child_links_recursive(self, url: str, visited: Set[str], *, depth: int=0) -> Iterator[Document]:
    """Recursively get all child links starting with the path of the input URL.

        Args:
            url: The URL to crawl.
            visited: A set of visited URLs.
            depth: Current depth of recursion. Stop when depth >= max_depth.
        """
    if depth >= self.max_depth:
        return
    visited.add(url)
    try:
        response = requests.get(url, timeout=self.timeout, headers=self.headers)
        if self.check_response_status and 400 <= response.status_code <= 599:
            raise ValueError(f'Received HTTP status {response.status_code}')
    except Exception as e:
        if self.continue_on_failure:
            logger.warning(f'Unable to load from {url}. Received error {e} of type {e.__class__.__name__}')
            return
        else:
            raise e
    content = self.extractor(response.text)
    if content:
        yield Document(page_content=content, metadata=self.metadata_extractor(response.text, url))
    sub_links = extract_sub_links(response.text, url, base_url=self.base_url, pattern=self.link_regex, prevent_outside=self.prevent_outside, exclude_prefixes=self.exclude_dirs, continue_on_failure=self.continue_on_failure)
    for link in sub_links:
        if link not in visited:
            yield from self._get_child_links_recursive(link, visited, depth=depth + 1)