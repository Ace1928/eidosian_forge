import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, AsyncIterator, Dict, Iterator, List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
class PlaywrightURLLoader(BaseLoader):
    """Load `HTML` pages with `Playwright` and parse with `Unstructured`.

    This is useful for loading pages that require javascript to render.

    Attributes:
        urls (List[str]): List of URLs to load.
        continue_on_failure (bool): If True, continue loading other URLs on failure.
        headless (bool): If True, the browser will run in headless mode.
        proxy (Optional[Dict[str, str]]): If set, the browser will access URLs
            through the specified proxy.

    Example:
        .. code-block:: python

            from langchain_community.document_loaders import PlaywrightURLLoader

            urls = ["https://api.ipify.org/?format=json",]
            proxy={
                "server": "https://xx.xx.xx:15818", # https://<host>:<port>
                "username": "username",
                "password": "password"
            }
            loader = PlaywrightURLLoader(urls, proxy=proxy)
            data = loader.load()
    """

    def __init__(self, urls: List[str], continue_on_failure: bool=True, headless: bool=True, remove_selectors: Optional[List[str]]=None, evaluator: Optional[PlaywrightEvaluator]=None, proxy: Optional[Dict[str, str]]=None):
        """Load a list of URLs using Playwright."""
        try:
            import playwright
        except ImportError:
            raise ImportError('playwright package not found, please install it with `pip install playwright`')
        self.urls = urls
        self.continue_on_failure = continue_on_failure
        self.headless = headless
        self.proxy = proxy
        if remove_selectors and evaluator:
            raise ValueError('`remove_selectors` and `evaluator` cannot be both not None')
        self.evaluator = evaluator or UnstructuredHtmlEvaluator(remove_selectors)

    def lazy_load(self) -> Iterator[Document]:
        """Load the specified URLs using Playwright and create Document instances.

        Returns:
            A list of Document instances with loaded content.
        """
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless, proxy=self.proxy)
            for url in self.urls:
                try:
                    page = browser.new_page()
                    response = page.goto(url)
                    if response is None:
                        raise ValueError(f'page.goto() returned None for url {url}')
                    text = self.evaluator.evaluate(page, browser, response)
                    metadata = {'source': url}
                    yield Document(page_content=text, metadata=metadata)
                except Exception as e:
                    if self.continue_on_failure:
                        logger.error(f'Error fetching or processing {url}, exception: {e}')
                    else:
                        raise e
            browser.close()

    async def aload(self) -> List[Document]:
        """Load the specified URLs with Playwright and create Documents asynchronously.
        Use this function when in a jupyter notebook environment.

        Returns:
            A list of Document instances with loaded content.
        """
        return [doc async for doc in self.alazy_load()]

    async def alazy_load(self) -> AsyncIterator[Document]:
        """Load the specified URLs with Playwright and create Documents asynchronously.
        Use this function when in a jupyter notebook environment.

        Returns:
            A list of Document instances with loaded content.
        """
        from playwright.async_api import async_playwright
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.headless, proxy=self.proxy)
            for url in self.urls:
                try:
                    page = await browser.new_page()
                    response = await page.goto(url)
                    if response is None:
                        raise ValueError(f'page.goto() returned None for url {url}')
                    text = await self.evaluator.evaluate_async(page, browser, response)
                    metadata = {'source': url}
                    yield Document(page_content=text, metadata=metadata)
                except Exception as e:
                    if self.continue_on_failure:
                        logger.error(f'Error fetching or processing {url}, exception: {e}')
                    else:
                        raise e
            await browser.close()