import os
import re
import glob
import requests
import logging
from typing import Dict, Optional, List, Tuple
from ray._private.accelerators.accelerator import AcceleratorManager
def _get_tpu_metadata(key: str) -> Optional[str]:
    """Poll and get TPU metadata."""
    try:
        accelerator_type_request = requests.get(os.path.join(GCE_TPU_ACCELERATOR_ENDPOINT, key), headers=GCE_TPU_HEADERS)
        if accelerator_type_request.status_code == 200 and accelerator_type_request.text:
            return accelerator_type_request.text
        else:
            logging.debug(f'Unable to poll TPU GCE Metadata. Got status code: {accelerator_type_request.status_code} and content: {accelerator_type_request.text}')
    except requests.RequestException as e:
        logging.debug('Unable to poll the TPU GCE Metadata: %s', e)
    return None