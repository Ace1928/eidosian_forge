from __future__ import annotations
import time
from ..types.properties import StatefulProperty
from ..types.auth import APIKeyData
from .admin import AZManagementClient, logger
from ..utils.lazy import get_az_flow
from typing import List, Optional, Any, TYPE_CHECKING

            Fetches the API Key Data
            