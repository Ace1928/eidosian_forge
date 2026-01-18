import json
import logging
import os
import re
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
from langchain_core.output_parsers.transform import BaseOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult, Generation
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
class KineticaUtil:
    """Kinetica utility functions."""

    @classmethod
    def create_kdbc(cls, url: Optional[str]=None, user: Optional[str]=None, passwd: Optional[str]=None) -> 'gpudb.GPUdb':
        """Create a connectica connection object and verify connectivity.

        If None is passed for one or more of the parameters then an attempt will be made
        to retrieve the value from the related environment variable.

        Args:
            url: The Kinetica URL or ``KINETICA_URL`` if None.
            user: The Kinetica user or ``KINETICA_USER`` if None.
            passwd: The Kinetica password or ``KINETICA_PASSWD`` if None.

        Returns:
            The Kinetica connection object.
        """
        try:
            import gpudb
        except ModuleNotFoundError:
            raise ImportError('Could not import Kinetica python package. Please install it with `pip install gpudb`.')
        url = cls._get_env('KINETICA_URL', url)
        user = cls._get_env('KINETICA_USER', user)
        passwd = cls._get_env('KINETICA_PASSWD', passwd)
        options = gpudb.GPUdb.Options()
        options.username = user
        options.password = passwd
        options.skip_ssl_cert_verification = True
        options.disable_failover = True
        options.logging_level = 'INFO'
        kdbc = gpudb.GPUdb(host=url, options=options)
        LOG.info('Connected to Kinetica: {}. (api={}, server={})'.format(kdbc.get_url(), version('gpudb'), kdbc.server_version))
        return kdbc

    @classmethod
    def _get_env(cls, name: str, default: Optional[str]) -> str:
        """Get an environment variable or use a default."""
        if default is not None:
            return default
        result = os.getenv(name)
        if result is not None:
            return result
        raise ValueError(f'Parameter was not passed and not found in the environment: {name}')