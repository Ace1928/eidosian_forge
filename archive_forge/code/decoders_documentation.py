from __future__ import annotations
from pydantic import ValidationError
from lazyops.libs import lazyload
from lazyops.libs.pooler import ThreadPooler
from .lazy import get_az_settings, get_az_resource
from typing import Any, Optional, List

    Gets the Auth Token from the Headers
    