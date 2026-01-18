import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple
import wandb
from wandb.sdk.integration_utils.auto_logging import Response
from wandb.sdk.lib.runid import generate_id
def _resolve_chat_response(self, response: Response) -> List[Dict[str, Any]]:
    return [subset_dict(response.__dict__, ['response_id', 'generation_id', 'query', 'text', 'conversation_id', 'prompt', 'chatlog', 'preamble'])]