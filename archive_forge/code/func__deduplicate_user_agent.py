from typing import Dict, Optional, Union
from .. import constants
from ._runtime import (
from ._token import get_token
from ._validators import validate_hf_hub_args
def _deduplicate_user_agent(user_agent: str) -> str:
    """Deduplicate redundant information in the generated user-agent."""
    return '; '.join({key.strip(): None for key in user_agent.split(';')}.keys())