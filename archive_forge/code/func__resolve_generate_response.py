import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple
import wandb
from wandb.sdk.integration_utils.auto_logging import Response
from wandb.sdk.lib.runid import generate_id
def _resolve_generate_response(self, response: Response) -> List[Dict[str, Any]]:
    return_list = []
    for _response in response:
        _response_dict = _response._visualize_helper()
        try:
            _response_dict['token_likelihoods'] = wandb.Html(_response_dict['token_likelihoods'])
        except (KeyError, ValueError):
            pass
        return_list.append(_response_dict)
    return return_list