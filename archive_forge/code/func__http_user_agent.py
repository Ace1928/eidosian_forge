from typing import Dict, Optional, Union
from .. import constants
from ._runtime import (
from ._token import get_token
from ._validators import validate_hf_hub_args
def _http_user_agent(*, library_name: Optional[str]=None, library_version: Optional[str]=None, user_agent: Union[Dict, str, None]=None) -> str:
    """Format a user-agent string containing information about the installed packages.

    Args:
        library_name (`str`, *optional*):
            The name of the library that is making the HTTP request.
        library_version (`str`, *optional*):
            The version of the library that is making the HTTP request.
        user_agent (`str`, `dict`, *optional*):
            The user agent info in the form of a dictionary or a single string.

    Returns:
        The formatted user-agent string.
    """
    if library_name is not None:
        ua = f'{library_name}/{library_version}'
    else:
        ua = 'unknown/None'
    ua += f'; hf_hub/{get_hf_hub_version()}'
    ua += f'; python/{get_python_version()}'
    if not constants.HF_HUB_DISABLE_TELEMETRY:
        if is_torch_available():
            ua += f'; torch/{get_torch_version()}'
        if is_tf_available():
            ua += f'; tensorflow/{get_tf_version()}'
        if is_fastai_available():
            ua += f'; fastai/{get_fastai_version()}'
        if is_fastcore_available():
            ua += f'; fastcore/{get_fastcore_version()}'
    if isinstance(user_agent, dict):
        ua += '; ' + '; '.join((f'{k}/{v}' for k, v in user_agent.items()))
    elif isinstance(user_agent, str):
        ua += '; ' + user_agent
    return _deduplicate_user_agent(ua)