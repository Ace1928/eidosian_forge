from ._base import *
from .models import Params, component_name
from .exceptions import *
from .dependencies import fix_query_dependencies, clone_dependant, insert_dependencies
class RequestShadow(Request):

    def __init__(self, request: Request):
        super().__init__(scope=ChainMap({}, request.scope))
        self.request = request

    async def stream(self):
        async for body in self.request.stream():
            yield body

    async def body(self):
        return await self.request.body()

    async def json(self):
        return await self.request.json()

    async def form(self):
        return await self.request.form()

    async def close(self):
        raise NotImplementedError

    async def is_disconnected(self):
        return await self.request.is_disconnected()