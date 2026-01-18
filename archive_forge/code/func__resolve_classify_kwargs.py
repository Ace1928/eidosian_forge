import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple
import wandb
from wandb.sdk.integration_utils.auto_logging import Response
from wandb.sdk.lib.runid import generate_id
def _resolve_classify_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    example_texts = []
    example_labels = []
    for example in kwargs['examples']:
        example_texts.append(example.text)
        example_labels.append(example.label)
    kwargs.pop('examples')
    kwargs['example_texts'] = example_texts
    kwargs['example_labels'] = example_labels
    return kwargs