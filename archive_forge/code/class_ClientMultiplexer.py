import asyncio
import threading
from typing import Generic, TypeVar, Callable, Dict, Awaitable
class ClientMultiplexer(Generic[_Key, _Client]):
    _OpenedClientFactory = Callable[[_Key], _Client]
    _ClientCloser = Callable[[_Client], None]
    _factory: _OpenedClientFactory
    _closer: _ClientCloser
    _lock: threading.Lock
    _live_clients: Dict[_Key, _Client]

    def __init__(self, factory: _OpenedClientFactory, closer: _ClientCloser=lambda client: client.__exit__(None, None, None)):
        self._factory = factory
        self._closer = closer
        self._lock = threading.Lock()
        self._live_clients = {}

    def get_or_create(self, key: _Key) -> _Client:
        with self._lock:
            if key not in self._live_clients:
                self._live_clients[key] = self._factory(key)
            return self._live_clients[key]

    def try_erase(self, key: _Key, client: _Client):
        with self._lock:
            if key not in self._live_clients:
                return
            current_client = self._live_clients[key]
            if current_client is not client:
                return
            del self._live_clients[key]
        self._closer(client)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        live_clients: Dict[_Key, _Client]
        with self._lock:
            live_clients = self._live_clients
            self._live_clients = {}
        for topic, client in live_clients.items():
            self._closer(client)