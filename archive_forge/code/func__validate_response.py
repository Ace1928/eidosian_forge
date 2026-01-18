from __future__ import annotations
import inspect
from abc import ABC
from urllib.parse import urljoin
from fastapi import HTTPException
from lazyops.libs import lazyload
from lazyops.libs.proxyobj import ProxyObject
from lazyops.utils.helpers import timed_cache
from ..types.errors import InvalidOperationException
from ..types.auth import AuthZeroTokenAuth
from ..types.clients import AuthZeroClientObject
from ..utils.lazy import get_az_settings, logger
from .tokens import ClientCredentialsFlow
from typing import Optional, List, Dict, Any, Union
def _validate_response(self, response: Union['Response', 'AsyncResponse']):
    """
        Validates the Response
        """
    try:
        response.raise_for_status()
    except niquests.HTTPError as e:
        if e.response.status_code in {400, 404}:
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text) from e
        operation_name = f'{self.__class__.__name__}.{inspect.stack()[1].function}'
        msg = f'[{response.status_code}] Error doing `{operation_name}`'
        if hasattr(self, 'user_id'):
            msg += f' on user {self.user_id}'
        msg += f': {e.response.text}'
        logger.error(msg)
        raise InvalidOperationException(detail=msg) from e