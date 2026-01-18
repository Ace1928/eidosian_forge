import base64
import logging
import cloudpickle
from threading import Thread
from typing import Any, Optional, Tuple, Dict, List
import requests  # type: ignore
from fugue.rpc.base import RPCClient, RPCServer
from triad.utils.convert import to_timedelta
from werkzeug.serving import make_server
from flask import Flask, request
class FlaskRPCClient(RPCClient):
    """Flask RPC Client that can be used distributedly.
    Use :meth:`~.FlaskRPCServer.make_client` to create this instance.

    :param key: the unique key for the handler and this client
    :param host: the host address of the flask server
    :param port: the port of the flask server
    :param timeout_sec: timeout seconds for flask clients
    """

    def __init__(self, key: str, host: str, port: int, timeout_sec: float):
        self._url = f'http://{host}:{port}/invoke'
        self._timeout_sec = timeout_sec
        self._key = key

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the underlying function on the server side"""
        timeout: Any = None if self._timeout_sec <= 0 else self._timeout_sec
        res = requests.post(self._url, data=dict(key=self._key, value=_encode(*args, **kwargs)), timeout=timeout)
        res.raise_for_status()
        return _decode(res.text)[0][0]