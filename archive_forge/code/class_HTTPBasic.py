import binascii
from base64 import b64decode
from typing import Optional
from fastapi.exceptions import HTTPException
from fastapi.openapi.models import HTTPBase as HTTPBaseModel
from fastapi.openapi.models import HTTPBearer as HTTPBearerModel
from fastapi.security.base import SecurityBase
from fastapi.security.utils import get_authorization_scheme_param
from pydantic import BaseModel
from starlette.requests import Request
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_403_FORBIDDEN
from typing_extensions import Annotated, Doc  # type: ignore [attr-defined]
class HTTPBasic(HTTPBase):
    """
    HTTP Basic authentication.

    ## Usage

    Create an instance object and use that object as the dependency in `Depends()`.

    The dependency result will be an `HTTPBasicCredentials` object containing the
    `username` and the `password`.

    Read more about it in the
    [FastAPI docs for HTTP Basic Auth](https://fastapi.tiangolo.com/advanced/security/http-basic-auth/).

    ## Example

    ```python
    from typing import Annotated

    from fastapi import Depends, FastAPI
    from fastapi.security import HTTPBasic, HTTPBasicCredentials

    app = FastAPI()

    security = HTTPBasic()


    @app.get("/users/me")
    def read_current_user(credentials: Annotated[HTTPBasicCredentials, Depends(security)]):
        return {"username": credentials.username, "password": credentials.password}
    ```
    """

    def __init__(self, *, scheme_name: Annotated[Optional[str], Doc('\n                Security scheme name.\n\n                It will be included in the generated OpenAPI (e.g. visible at `/docs`).\n                ')]=None, realm: Annotated[Optional[str], Doc('\n                HTTP Basic authentication realm.\n                ')]=None, description: Annotated[Optional[str], Doc('\n                Security scheme description.\n\n                It will be included in the generated OpenAPI (e.g. visible at `/docs`).\n                ')]=None, auto_error: Annotated[bool, Doc('\n                By default, if the HTTP Basic authentication is not provided (a\n                header), `HTTPBasic` will automatically cancel the request and send the\n                client an error.\n\n                If `auto_error` is set to `False`, when the HTTP Basic authentication\n                is not available, instead of erroring out, the dependency result will\n                be `None`.\n\n                This is useful when you want to have optional authentication.\n\n                It is also useful when you want to have authentication that can be\n                provided in one of multiple optional ways (for example, in HTTP Basic\n                authentication or in an HTTP Bearer token).\n                ')]=True):
        self.model = HTTPBaseModel(scheme='basic', description=description)
        self.scheme_name = scheme_name or self.__class__.__name__
        self.realm = realm
        self.auto_error = auto_error

    async def __call__(self, request: Request) -> Optional[HTTPBasicCredentials]:
        authorization = request.headers.get('Authorization')
        scheme, param = get_authorization_scheme_param(authorization)
        if self.realm:
            unauthorized_headers = {'WWW-Authenticate': f'Basic realm="{self.realm}"'}
        else:
            unauthorized_headers = {'WWW-Authenticate': 'Basic'}
        if not authorization or scheme.lower() != 'basic':
            if self.auto_error:
                raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail='Not authenticated', headers=unauthorized_headers)
            else:
                return None
        invalid_user_credentials_exc = HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail='Invalid authentication credentials', headers=unauthorized_headers)
        try:
            data = b64decode(param).decode('ascii')
        except (ValueError, UnicodeDecodeError, binascii.Error):
            raise invalid_user_credentials_exc
        username, separator, password = data.partition(':')
        if not separator:
            raise invalid_user_credentials_exc
        return HTTPBasicCredentials(username=username, password=password)