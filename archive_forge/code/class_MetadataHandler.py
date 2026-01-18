from __future__ import annotations
import logging # isort:skip
import json
from tornado.web import authenticated
from .auth_request_handler import AuthRequestHandler
from .session_handler import SessionHandler
class MetadataHandler(SessionHandler, AuthRequestHandler):
    """ Implements a custom Tornado handler for document display page

    """

    @authenticated
    async def get(self, *args, **kwargs):
        url = self.application_context.url
        userdata = self.application_context.application.metadata
        if callable(userdata):
            userdata = userdata()
        if userdata is None:
            userdata = {}
        metadata = dict(url=url, data=userdata)
        self.set_header('Content-Type', 'application/json')
        self.write(json.dumps(metadata))