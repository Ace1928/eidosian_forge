from __future__ import annotations
import typing as t
from jinja2.loaders import BaseLoader
from werkzeug.routing import RequestRedirect
from .blueprints import Blueprint
from .globals import request_ctx
from .sansio.app import App
class newcls(oldcls):

    def __getitem__(self, key: str) -> t.Any:
        try:
            return super().__getitem__(key)
        except KeyError as e:
            if key not in request.form:
                raise
            raise DebugFilesKeyError(request, key).with_traceback(e.__traceback__) from None