from __future__ import annotations
import json
from logging import Logger
import requests
import tornado
from jupyter_server.base.handlers import APIHandler
class ListingsHandler(APIHandler):
    """An handler that returns the listings specs."""
    'Below fields are class level fields that are accessed and populated\n    by the initialization and the fetch_listings methods.\n    Some fields are initialized before the handler creation in the\n    handlers.py#add_handlers method.\n    Having those fields predefined reduces the guards in the methods using\n    them.\n    '
    blocked_extensions_uris: set = set()
    allowed_extensions_uris: set = set()
    blocked_extensions: list = []
    allowed_extensions: list = []
    listings_request_opts: dict = {}
    listings_refresh_seconds: int
    pc = None

    def get(self, path: str) -> None:
        """Get the listings for the extension manager."""
        self.set_header('Content-Type', 'application/json')
        if path == LISTINGS_URL_SUFFIX:
            self.write(ListingsHandler.listings)
        else:
            raise tornado.web.HTTPError(400)