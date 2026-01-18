from __future__ import annotations
import sys
import traceback
from collections import abc
from typing import (
from bson.son import SON
from pymongo import ASCENDING
from pymongo.errors import (
from pymongo.hello import HelloCompat
def _get_wce_doc(result: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    """Return the writeConcernError or None."""
    wce = result.get('writeConcernError')
    if wce:
        error_labels = result.get('errorLabels')
        if error_labels:
            wce['errorLabels'] = error_labels
    return wce