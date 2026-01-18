import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, AsyncIterator, Dict, Iterator, List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
class UnstructuredHtmlEvaluator(PlaywrightEvaluator):
    """Evaluates the page HTML content using the `unstructured` library."""

    def __init__(self, remove_selectors: Optional[List[str]]=None):
        """Initialize UnstructuredHtmlEvaluator."""
        try:
            import unstructured
        except ImportError:
            raise ImportError('unstructured package not found, please install it with `pip install unstructured`')
        self.remove_selectors = remove_selectors

    def evaluate(self, page: 'Page', browser: 'Browser', response: 'Response') -> str:
        """Synchronously process the HTML content of the page."""
        from unstructured.partition.html import partition_html
        for selector in self.remove_selectors or []:
            elements = page.locator(selector).all()
            for element in elements:
                if element.is_visible():
                    element.evaluate('element => element.remove()')
        page_source = page.content()
        elements = partition_html(text=page_source)
        return '\n\n'.join([str(el) for el in elements])

    async def evaluate_async(self, page: 'AsyncPage', browser: 'AsyncBrowser', response: 'AsyncResponse') -> str:
        """Asynchronously process the HTML content of the page."""
        from unstructured.partition.html import partition_html
        for selector in self.remove_selectors or []:
            elements = await page.locator(selector).all()
            for element in elements:
                if await element.is_visible():
                    await element.evaluate('element => element.remove()')
        page_source = await page.content()
        elements = partition_html(text=page_source)
        return '\n\n'.join([str(el) for el in elements])