from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from fastapi import params
from fastapi._compat import Undefined
from fastapi.openapi.models import Example
from typing_extensions import Annotated, Doc, deprecated  # type: ignore [attr-defined]
def Depends(dependency: Annotated[Optional[Callable[..., Any]], Doc('\n            A "dependable" callable (like a function).\n\n            Don\'t call it directly, FastAPI will call it for you, just pass the object\n            directly.\n            ')]=None, *, use_cache: Annotated[bool, Doc('\n            By default, after a dependency is called the first time in a request, if\n            the dependency is declared again for the rest of the request (for example\n            if the dependency is needed by several dependencies), the value will be\n            re-used for the rest of the request.\n\n            Set `use_cache` to `False` to disable this behavior and ensure the\n            dependency is called again (if declared more than once) in the same request.\n            ')]=True) -> Any:
    """
    Declare a FastAPI dependency.

    It takes a single "dependable" callable (like a function).

    Don't call it directly, FastAPI will call it for you.

    Read more about it in the
    [FastAPI docs for Dependencies](https://fastapi.tiangolo.com/tutorial/dependencies/).

    **Example**

    ```python
    from typing import Annotated

    from fastapi import Depends, FastAPI

    app = FastAPI()


    async def common_parameters(q: str | None = None, skip: int = 0, limit: int = 100):
        return {"q": q, "skip": skip, "limit": limit}


    @app.get("/items/")
    async def read_items(commons: Annotated[dict, Depends(common_parameters)]):
        return commons
    ```
    """
    return params.Depends(dependency=dependency, use_cache=use_cache)