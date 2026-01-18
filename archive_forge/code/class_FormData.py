from __future__ import annotations
import typing
from shlex import shlex
from urllib.parse import SplitResult, parse_qsl, urlencode, urlsplit
from starlette.concurrency import run_in_threadpool
from starlette.types import Scope
class FormData(ImmutableMultiDict[str, typing.Union[UploadFile, str]]):
    """
    An immutable multidict, containing both file uploads and text input.
    """

    def __init__(self, *args: FormData | typing.Mapping[str, str | UploadFile] | list[tuple[str, str | UploadFile]], **kwargs: str | UploadFile) -> None:
        super().__init__(*args, **kwargs)

    async def close(self) -> None:
        for key, value in self.multi_items():
            if isinstance(value, UploadFile):
                await value.close()