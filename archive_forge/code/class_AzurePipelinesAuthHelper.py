from __future__ import annotations
import os
import tempfile
import uuid
import typing as t
import urllib.parse
from ..encoding import (
from ..config import (
from ..git import (
from ..http import (
from ..util import (
from . import (
class AzurePipelinesAuthHelper(CryptographyAuthHelper):
    """
    Authentication helper for Azure Pipelines.
    Based on cryptography since it is provided by the default Azure Pipelines environment.
    """

    def publish_public_key(self, public_key_pem: str) -> None:
        """Publish the given public key."""
        try:
            agent_temp_directory = os.environ['AGENT_TEMPDIRECTORY']
        except KeyError as ex:
            raise MissingEnvironmentVariable(name=ex.args[0]) from None
        with tempfile.NamedTemporaryFile(prefix='public-key-', suffix='.pem', delete=False, dir=agent_temp_directory) as public_key_file:
            public_key_file.write(to_bytes(public_key_pem))
            public_key_file.flush()
        vso_add_attachment('ansible-core-ci', 'public-key.pem', public_key_file.name)