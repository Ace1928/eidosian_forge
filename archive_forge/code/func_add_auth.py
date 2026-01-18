import base64
import json
import logging
import os
import platform
from typing import Any, Dict, Mapping, Optional, Tuple, Union
import dockerpycreds  # type: ignore
def add_auth(self, reg: str, data: Dict[str, Any]) -> None:
    self['auths'][reg] = data