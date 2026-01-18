from typing import Any, Dict, List, Optional, Union, cast
from fastapi.exceptions import HTTPException
from fastapi.openapi.models import OAuth2 as OAuth2Model
from fastapi.openapi.models import OAuthFlows as OAuthFlowsModel
from fastapi.param_functions import Form
from fastapi.security.base import SecurityBase
from fastapi.security.utils import get_authorization_scheme_param
from starlette.requests import Request
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_403_FORBIDDEN
from typing_extensions import Annotated, Doc  # type: ignore [attr-defined]
class SecurityScopes:
    """
    This is a special class that you can define in a parameter in a dependency to
    obtain the OAuth2 scopes required by all the dependencies in the same chain.

    This way, multiple dependencies can have different scopes, even when used in the
    same *path operation*. And with this, you can access all the scopes required in
    all those dependencies in a single place.

    Read more about it in the
    [FastAPI docs for OAuth2 scopes](https://fastapi.tiangolo.com/advanced/security/oauth2-scopes/).
    """

    def __init__(self, scopes: Annotated[Optional[List[str]], Doc('\n                This will be filled by FastAPI.\n                ')]=None):
        self.scopes: Annotated[List[str], Doc('\n                The list of all the scopes required by dependencies.\n                ')] = scopes or []
        self.scope_str: Annotated[str, Doc('\n                All the scopes required by all the dependencies in a single string\n                separated by spaces, as defined in the OAuth2 specification.\n                ')] = ' '.join(self.scopes)