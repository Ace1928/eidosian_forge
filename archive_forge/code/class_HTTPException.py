from typing import Any, Dict, Optional, Sequence, Type, Union
from pydantic import BaseModel, create_model
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.exceptions import WebSocketException as StarletteWebSocketException
from typing_extensions import Annotated, Doc  # type: ignore [attr-defined]
class HTTPException(StarletteHTTPException):
    """
    An HTTP exception you can raise in your own code to show errors to the client.

    This is for client errors, invalid authentication, invalid data, etc. Not for server
    errors in your code.

    Read more about it in the
    [FastAPI docs for Handling Errors](https://fastapi.tiangolo.com/tutorial/handling-errors/).

    ## Example

    ```python
    from fastapi import FastAPI, HTTPException

    app = FastAPI()

    items = {"foo": "The Foo Wrestlers"}


    @app.get("/items/{item_id}")
    async def read_item(item_id: str):
        if item_id not in items:
            raise HTTPException(status_code=404, detail="Item not found")
        return {"item": items[item_id]}
    ```
    """

    def __init__(self, status_code: Annotated[int, Doc('\n                HTTP status code to send to the client.\n                ')], detail: Annotated[Any, Doc('\n                Any data to be sent to the client in the `detail` key of the JSON\n                response.\n                ')]=None, headers: Annotated[Optional[Dict[str, str]], Doc('\n                Any headers to send to the client in the response.\n                ')]=None) -> None:
        super().__init__(status_code=status_code, detail=detail, headers=headers)