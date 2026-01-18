from __future__ import annotations
import logging
import os
import sys
import typing as t
from datetime import timedelta
from itertools import chain
from werkzeug.exceptions import Aborter
from werkzeug.exceptions import BadRequest
from werkzeug.exceptions import BadRequestKeyError
from werkzeug.routing import BuildError
from werkzeug.routing import Map
from werkzeug.routing import Rule
from werkzeug.sansio.response import Response
from werkzeug.utils import cached_property
from werkzeug.utils import redirect as _wz_redirect
from .. import typing as ft
from ..config import Config
from ..config import ConfigAttribute
from ..ctx import _AppCtxGlobals
from ..helpers import _split_blueprint_path
from ..helpers import get_debug_flag
from ..json.provider import DefaultJSONProvider
from ..json.provider import JSONProvider
from ..logging import create_logger
from ..templating import DispatchingJinjaLoader
from ..templating import Environment
from .scaffold import _endpoint_from_view_func
from .scaffold import find_package
from .scaffold import Scaffold
from .scaffold import setupmethod
def _find_error_handler(self, e: Exception, blueprints: list[str]) -> ft.ErrorHandlerCallable | None:
    """Return a registered error handler for an exception in this order:
        blueprint handler for a specific code, app handler for a specific code,
        blueprint handler for an exception class, app handler for an exception
        class, or ``None`` if a suitable handler is not found.
        """
    exc_class, code = self._get_exc_class_and_code(type(e))
    names = (*blueprints, None)
    for c in (code, None) if code is not None else (None,):
        for name in names:
            handler_map = self.error_handler_spec[name][c]
            if not handler_map:
                continue
            for cls in exc_class.__mro__:
                handler = handler_map.get(cls)
                if handler is not None:
                    return handler
    return None