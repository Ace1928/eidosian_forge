from typing import Any, Dict, Optional, Sequence, Type, Union
from pydantic import BaseModel, create_model
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.exceptions import WebSocketException as StarletteWebSocketException
from typing_extensions import Annotated, Doc  # type: ignore [attr-defined]
class ResponseValidationError(ValidationException):

    def __init__(self, errors: Sequence[Any], *, body: Any=None) -> None:
        super().__init__(errors)
        self.body = body

    def __str__(self) -> str:
        message = f'{len(self._errors)} validation errors:\n'
        for err in self._errors:
            message += f'  {err}\n'
        return message