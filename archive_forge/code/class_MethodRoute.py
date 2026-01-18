from ._base import *
from .models import Params, component_name
from .exceptions import *
from .dependencies import fix_query_dependencies, clone_dependant, insert_dependencies
class MethodRoute(APIRoute):

    def __init__(self, entrypoint: 'Entrypoint', path: str, func: Union[FunctionType, CoroutineType], *, result_model: Type[Any]=None, name: str=None, errors: Sequence[Type[BaseError]]=None, dependencies: Sequence[Depends]=None, response_class: Type[Response]=JSONResponse, request_class: Type[JsonRpcRequest]=JsonRpcRequest, middlewares: Sequence[JsonRpcMiddleware]=None, **kwargs):
        name = name or func.__name__
        result_model = result_model or func.__annotations__.get('return')
        _, path_format, _ = compile_path(path)
        func_dependant = get_dependant(path=path_format, call=func)
        insert_dependencies(func_dependant, dependencies)
        insert_dependencies(func_dependant, entrypoint.common_dependencies)
        fix_query_dependencies(func_dependant)
        flat_dependant = get_flat_dependant(func_dependant, skip_repeats=True)
        _Request = make_request_model(name, func.__module__, flat_dependant.body_params)

        @component_name(f'_Response[{name}]', func.__module__)
        class _Response(BaseModel):
            jsonrpc: StrictStr = Field('2.0', const=True, example='2.0')
            id: Union[StrictStr, int] = Field(None, example=0)
            result: result_model

            class Config:
                extra = 'forbid'

        async def endpoint(__request__: _Request):
            del __request__
        endpoint.__name__ = func.__name__
        endpoint.__doc__ = func.__doc__
        responses = errors_responses(errors)
        super().__init__(path, endpoint, methods=['POST'], name=name, response_class=response_class, response_model=_Response, responses=responses, **kwargs)
        self.dependant.path_params = func_dependant.path_params
        self.dependant.header_params = func_dependant.header_params
        self.dependant.cookie_params = func_dependant.cookie_params
        self.dependant.dependencies = func_dependant.dependencies
        self.dependant.security_requirements = func_dependant.security_requirements
        self.func = func
        self.func_dependant = func_dependant
        self.entrypoint = entrypoint
        self.middlewares = middlewares or []
        self.app = request_response(self.handle_http_request)
        self.request_class = request_class

    async def parse_body(self, http_request) -> Any:
        try:
            req = await http_request.json()
        except JSONDecodeError:
            raise ParseError()
        return req

    async def handle_http_request(self, http_request: Request):
        background_tasks = BackgroundTasks()
        sub_response = Response(content=None, status_code=None, headers=None, media_type=None, background=None)
        try:
            body = await self.parse_body(http_request)
        except Exception as exc:
            resp, _ = await self.entrypoint.handle_exception_to_resp(exc)
            response = self.response_class(content=resp, background=background_tasks)
        else:
            try:
                resp = await self.handle_body(http_request, background_tasks, sub_response, body)
            except NoContent:
                response = Response(media_type='application/json', background=background_tasks)
            else:
                response = self.response_class(content=resp, background=background_tasks)
        response.headers.raw.extend(sub_response.headers.raw)
        if sub_response.status_code:
            response.status_code = sub_response.status_code
        return response

    async def handle_body(self, http_request: Request, background_tasks: BackgroundTasks, sub_response: Response, body: Any) -> dict:
        shared_dependencies_error = None
        try:
            dependency_cache = await self.entrypoint.solve_shared_dependencies(http_request, background_tasks, sub_response)
        except BaseError as error:
            shared_dependencies_error = error
            dependency_cache = None
        resp = await self.handle_req_to_resp(http_request, background_tasks, sub_response, body, dependency_cache=dependency_cache, shared_dependencies_error=shared_dependencies_error)
        has_content = 'error' in resp or 'id' in resp
        if not has_content:
            raise NoContent
        return resp

    async def handle_req_to_resp(self, http_request: Request, background_tasks: BackgroundTasks, sub_response: Response, req: Any, dependency_cache: dict=None, shared_dependencies_error: BaseError=None) -> dict:
        async with JsonRpcContext(entrypoint=self.entrypoint, method_route=self, raw_request=req, http_request=http_request, background_tasks=background_tasks, http_response=sub_response, json_rpc_request_class=self.request_class) as ctx:
            await ctx.enter_middlewares(self.entrypoint.middlewares)
            if ctx.request.method != self.name:
                raise MethodNotFound
            resp = await self.handle_req(http_request, background_tasks, sub_response, ctx, dependency_cache=dependency_cache, shared_dependencies_error=shared_dependencies_error)
            ctx.on_raw_response(resp)
        return ctx.raw_response

    async def handle_req(self, http_request: Request, background_tasks: BackgroundTasks, sub_response: Response, ctx: JsonRpcContext, dependency_cache: dict=None, shared_dependencies_error: BaseError=None):
        await ctx.enter_middlewares(self.middlewares)
        if shared_dependencies_error:
            raise shared_dependencies_error
        dependency_cache = dependency_cache.copy()
        values, errors, background_tasks, _, _ = await solve_dependencies(request=http_request, dependant=self.func_dependant, body=ctx.request.params, background_tasks=background_tasks, response=sub_response, dependency_overrides_provider=self.dependency_overrides_provider, dependency_cache=dependency_cache)
        if errors:
            raise invalid_params_from_validation_error(RequestValidationError(errors))
        result = await call_sync_async(self.func, **values)
        response = {'jsonrpc': '2.0', 'result': result}
        resp = await serialize_response(field=self.secure_cloned_response_field, response_content=response, include=self.response_model_include, exclude=self.response_model_exclude, by_alias=self.response_model_by_alias, exclude_unset=self.response_model_exclude_unset)
        return resp