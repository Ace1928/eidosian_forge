from __future__ import annotations
import typing
from dataclasses import dataclass, field
from enum import Enum
from tempfile import SpooledTemporaryFile
from urllib.parse import unquote_plus
from starlette.datastructures import FormData, Headers, UploadFile
class FormParser:

    def __init__(self, headers: Headers, stream: typing.AsyncGenerator[bytes, None]) -> None:
        assert multipart is not None, 'The `python-multipart` library must be installed to use form parsing.'
        self.headers = headers
        self.stream = stream
        self.messages: list[tuple[FormMessage, bytes]] = []

    def on_field_start(self) -> None:
        message = (FormMessage.FIELD_START, b'')
        self.messages.append(message)

    def on_field_name(self, data: bytes, start: int, end: int) -> None:
        message = (FormMessage.FIELD_NAME, data[start:end])
        self.messages.append(message)

    def on_field_data(self, data: bytes, start: int, end: int) -> None:
        message = (FormMessage.FIELD_DATA, data[start:end])
        self.messages.append(message)

    def on_field_end(self) -> None:
        message = (FormMessage.FIELD_END, b'')
        self.messages.append(message)

    def on_end(self) -> None:
        message = (FormMessage.END, b'')
        self.messages.append(message)

    async def parse(self) -> FormData:
        callbacks = {'on_field_start': self.on_field_start, 'on_field_name': self.on_field_name, 'on_field_data': self.on_field_data, 'on_field_end': self.on_field_end, 'on_end': self.on_end}
        parser = multipart.QuerystringParser(callbacks)
        field_name = b''
        field_value = b''
        items: list[tuple[str, typing.Union[str, UploadFile]]] = []
        async for chunk in self.stream:
            if chunk:
                parser.write(chunk)
            else:
                parser.finalize()
            messages = list(self.messages)
            self.messages.clear()
            for message_type, message_bytes in messages:
                if message_type == FormMessage.FIELD_START:
                    field_name = b''
                    field_value = b''
                elif message_type == FormMessage.FIELD_NAME:
                    field_name += message_bytes
                elif message_type == FormMessage.FIELD_DATA:
                    field_value += message_bytes
                elif message_type == FormMessage.FIELD_END:
                    name = unquote_plus(field_name.decode('latin-1'))
                    value = unquote_plus(field_value.decode('latin-1'))
                    items.append((name, value))
        return FormData(items)