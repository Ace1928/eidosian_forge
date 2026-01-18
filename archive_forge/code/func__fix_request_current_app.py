import re
from typing import TYPE_CHECKING, Tuple, Type, TypeVar
from .typedefs import Handler, Middleware
from .web_exceptions import HTTPMove, HTTPPermanentRedirect
from .web_request import Request
from .web_response import StreamResponse
from .web_urldispatcher import SystemRoute
def _fix_request_current_app(app: 'Application') -> Middleware:

    @middleware
    async def impl(request: Request, handler: Handler) -> StreamResponse:
        with request.match_info.set_current_app(app):
            return await handler(request)
    return impl