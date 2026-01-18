from typing import Iterator, Literal, Optional
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.utils import get_from_env
Initialize with API key and url.

        Args:
            url: The url to be crawled.
            api_key: The Firecrawl API key. If not specified will be read from env var
                FIREWALL_API_KEY. Get an API key
            mode: The mode to run the loader in. Default is "crawl".
                 Options include "scrape" (single url) and
                 "crawl" (all accessible sub pages).
            params: The parameters to pass to the Firecrawl API.
                Examples include crawlerOptions.
                For more details, visit: https://github.com/mendableai/firecrawl-py
        