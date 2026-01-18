import os
import platform
from copy import copy
from string import ascii_letters, digits
from typing import NamedTuple, Optional
from botocore import __version__ as botocore_version
from botocore.compat import HAS_CRT
def _build_legacy_ua_string(self, config_ua_override):
    components = [config_ua_override]
    if self._session_user_agent_extra:
        components.append(self._session_user_agent_extra)
    if self._client_config.user_agent_extra:
        components.append(self._client_config.user_agent_extra)
    return ' '.join(components)