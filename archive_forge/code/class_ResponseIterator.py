from collections import deque
import string
from typing import Deque, Union
import proto
import requests
import cloudsdk.google.protobuf.message
from cloudsdk.google.protobuf.json_format import Parse
class ResponseIterator:
    """Iterator over REST API responses.

    Args:
        response (requests.Response): An API response object.
        response_message_cls (Union[proto.Message, google.protobuf.message.Message]): A response
        class expected to be returned from an API.

    Raises:
        ValueError: If `response_message_cls` is not a subclass of `proto.Message` or `google.protobuf.message.Message`.
    """

    def __init__(self, response: requests.Response, response_message_cls: Union[proto.Message, cloudsdk.google.protobuf.message.Message]):
        self._response = response
        self._response_message_cls = response_message_cls
        self._response_itr = self._response.iter_content(decode_unicode=True)
        self._ready_objs: Deque[str] = deque()
        self._obj = ''
        self._level = 0
        self._in_string = False
        self._escape_next = False

    def cancel(self):
        """Cancel existing streaming operation."""
        self._response.close()

    def _process_chunk(self, chunk: str):
        if self._level == 0:
            if chunk[0] != '[':
                raise ValueError('Can only parse array of JSON objects, instead got %s' % chunk)
        for char in chunk:
            if char == '{':
                if self._level == 1:
                    self._obj = ''
                if not self._in_string:
                    self._level += 1
                self._obj += char
            elif char == '}':
                self._obj += char
                if not self._in_string:
                    self._level -= 1
                if not self._in_string and self._level == 1:
                    self._ready_objs.append(self._obj)
            elif char == '"':
                if not self._escape_next:
                    self._in_string = not self._in_string
                self._obj += char
            elif char in string.whitespace:
                if self._in_string:
                    self._obj += char
            elif char == '[':
                if self._level == 0:
                    self._level += 1
                else:
                    self._obj += char
            elif char == ']':
                if self._level == 1:
                    self._level -= 1
                else:
                    self._obj += char
            else:
                self._obj += char
            self._escape_next = not self._escape_next if char == '\\' else False

    def __next__(self):
        while not self._ready_objs:
            try:
                chunk = next(self._response_itr)
                self._process_chunk(chunk)
            except StopIteration as e:
                if self._level > 0:
                    raise ValueError('Unfinished stream: %s' % self._obj)
                raise e
        return self._grab()

    def _grab(self):
        if issubclass(self._response_message_cls, proto.Message):
            return self._response_message_cls.from_json(self._ready_objs.popleft())
        elif issubclass(self._response_message_cls, cloudsdk.google.protobuf.message.Message):
            return Parse(self._ready_objs.popleft(), self._response_message_cls())
        else:
            raise ValueError('Response message class must be a subclass of proto.Message or google.protobuf.message.Message.')

    def __iter__(self):
        return self