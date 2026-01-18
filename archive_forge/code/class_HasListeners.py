import asyncio
import enum
import json
import pathlib
import re
import shutil
import subprocess
import sys
from functools import lru_cache
from typing import (
from traitlets import Any as Any_
from traitlets import Instance
from traitlets import List as List_
from traitlets import Unicode, default
from traitlets.config import LoggingConfigurable
class HasListeners:
    _listeners = {str(scope.value): [] for scope in MessageScope}
    log: Any = Instance('logging.Logger')

    @classmethod
    def register_message_listener(cls, scope: Text, language_server: Optional[Text]=None, method: Optional[Text]=None):
        """register a listener for language server protocol messages"""

        def inner(listener: 'HandlerListenerCallback') -> 'HandlerListenerCallback':
            cls.unregister_message_listener(listener)
            cls._listeners[scope].append(MessageListener(listener=listener, language_server=language_server, method=method))
            return listener
        return inner

    @classmethod
    def unregister_message_listener(cls, listener: 'HandlerListenerCallback'):
        """unregister a listener for language server protocol messages"""
        for scope in MessageScope:
            cls._listeners[str(scope.value)] = [lst for lst in cls._listeners[str(scope.value)] if lst.listener != listener]

    async def wait_for_listeners(self, scope: MessageScope, message_str: Text, language_server: Text) -> None:
        scope_val = str(scope.value)
        listeners = self._listeners[scope_val] + self._listeners[MessageScope.ALL.value]
        if listeners:
            message = json.loads(message_str)
            futures = [listener(scope_val, message=message, language_server=language_server, manager=cast('LanguageServerManagerAPI', self)) for listener in listeners if listener.wants(message, language_server)]
            if futures:
                await asyncio.gather(*futures)