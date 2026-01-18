import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple
import wandb
from wandb.sdk.integration_utils.auto_logging import Response
from wandb.sdk.lib.runid import generate_id
def _resolve_summarize_response(self, response: Response) -> List[Dict[str, Any]]:
    return [{'response_id': response.id, 'summary': response.summary}]