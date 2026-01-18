import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
class Script:
    """
    An executable Lua script object returned by ``register_script``
    """

    def __init__(self, registered_client, script):
        self.registered_client = registered_client
        self.script = script
        if isinstance(script, str):
            encoder = self.get_encoder()
            script = encoder.encode(script)
        self.sha = hashlib.sha1(script).hexdigest()

    def __call__(self, keys=[], args=[], client=None):
        """Execute the script, passing any required ``args``"""
        if client is None:
            client = self.registered_client
        args = tuple(keys) + tuple(args)
        from redis.client import Pipeline
        if isinstance(client, Pipeline):
            client.scripts.add(self)
        try:
            return client.evalsha(self.sha, len(keys), *args)
        except NoScriptError:
            self.sha = client.script_load(self.script)
            return client.evalsha(self.sha, len(keys), *args)

    def get_encoder(self):
        """Get the encoder to encode string scripts into bytes."""
        try:
            return self.registered_client.get_encoder()
        except AttributeError:
            return self.registered_client.connection_pool.get_encoder()