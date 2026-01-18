from __future__ import annotations
import logging # isort:skip
from tornado.web import HTTPError, RequestHandler
class IcoHandler(RequestHandler):
    """ Implements a custom Tornado request handler for favicon.ico
    files.

    """

    def initialize(self, *args, **kw):
        self.app = kw.get('app')

    async def get(self, *args, **kwargs):
        if self.app.icon is None:
            raise HTTPError(status_code=404)
        self.set_header('Content-Type', 'image/x-icon')
        self.write(self.app.icon)
        return self.flush()