from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from fastapi import params
from fastapi._compat import Undefined
from fastapi.openapi.models import Example
from typing_extensions import Annotated, Doc, deprecated  # type: ignore [attr-defined]
def Security(dependency: Annotated[Optional[Callable[..., Any]], Doc('\n            A "dependable" callable (like a function).\n\n            Don\'t call it directly, FastAPI will call it for you, just pass the object\n            directly.\n            ')]=None, *, scopes: Annotated[Optional[Sequence[str]], Doc('\n            OAuth2 scopes required for the *path operation* that uses this Security\n            dependency.\n\n            The term "scope" comes from the OAuth2 specification, it seems to be\n            intentionaly vague and interpretable. It normally refers to permissions,\n            in cases to roles.\n\n            These scopes are integrated with OpenAPI (and the API docs at `/docs`).\n            So they are visible in the OpenAPI specification.\n            )\n            ')]=None, use_cache: Annotated[bool, Doc('\n            By default, after a dependency is called the first time in a request, if\n            the dependency is declared again for the rest of the request (for example\n            if the dependency is needed by several dependencies), the value will be\n            re-used for the rest of the request.\n\n            Set `use_cache` to `False` to disable this behavior and ensure the\n            dependency is called again (if declared more than once) in the same request.\n            ')]=True) -> Any:
    """
    Declare a FastAPI Security dependency.

    The only difference with a regular dependency is that it can declare OAuth2
    scopes that will be integrated with OpenAPI and the automatic UI docs (by default
    at `/docs`).

    It takes a single "dependable" callable (like a function).

    Don't call it directly, FastAPI will call it for you.

    Read more about it in the
    [FastAPI docs for Security](https://fastapi.tiangolo.com/tutorial/security/) and
    in the
    [FastAPI docs for OAuth2 scopes](https://fastapi.tiangolo.com/advanced/security/oauth2-scopes/).

    **Example**

    ```python
    from typing import Annotated

    from fastapi import Depends, FastAPI

    from .db import User
    from .security import get_current_active_user

    app = FastAPI()

    @app.get("/users/me/items/")
    async def read_own_items(
        current_user: Annotated[User, Security(get_current_active_user, scopes=["items"])]
    ):
        return [{"item_id": "Foo", "owner": current_user.username}]
    ```
    """
    return params.Security(dependency=dependency, scopes=scopes, use_cache=use_cache)