import json
import os
import tempfile
import traceback
from runpy import run_path
from unittest.mock import MagicMock
from urllib.parse import parse_qs
import param
from tornado import web
from tornado.wsgi import WSGIContainer
from ..entry_points import entry_points_for
from .state import state
class ParamHandler(BaseHandler):

    def __init__(self, app, request, **kwargs):
        self.root = kwargs.pop('root', None)
        super().__init__(app, request, **kwargs)

    @classmethod
    def serialize(cls, parameterized, parameters):
        values = {p: getattr(parameterized, p) for p in parameters}
        return parameterized.param.serialize_parameters(values)

    @classmethod
    def deserialize(cls, parameterized, parameters):
        for p in parameters:
            if p not in parameterized.param:
                reason = f"'{p}' query parameter not recognized."
                raise HTTPError(reason=reason, status_code=400)
        return {p: parameterized.param.deserialize_value(p, v) for p, v in parameters.items()}

    async def get(self):
        path = self.request.path
        endpoint = path[path.index(self.root) + len(self.root):]
        parameterized, parameters, _ = state._rest_endpoints.get(endpoint, (None, None, None))
        if not parameterized:
            return
        args = parse_qs(self.request.query)
        params = self.deserialize(parameterized[0], args)
        parameterized[0].param.update(**params)
        self.set_header('Content-Type', 'application/json')
        self.write(self.serialize(parameterized[0], parameters))