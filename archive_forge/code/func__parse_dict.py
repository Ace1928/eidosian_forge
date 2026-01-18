import logging
import os
from packaging import version
from copy import deepcopy
from typing import TYPE_CHECKING, Dict, Optional
from ray._private.dict import flatten_dict
def _parse_dict(self, dict_to_log: Dict) -> Dict:
    """Parses provided dict to convert all values to float.

        MLflow can only log metrics that are floats. This does not apply to
        logging parameters or artifacts.

        Args:
            dict_to_log: The dictionary containing the metrics to log.

        Returns:
            A dictionary containing the metrics to log with all values being
                converted to floats, or skipped if not able to be converted.
        """
    new_dict = {}
    for key, value in dict_to_log.items():
        try:
            value = float(value)
            new_dict[key] = value
        except (ValueError, TypeError):
            logger.debug('Cannot log key {} with value {} since the value cannot be converted to float.'.format(key, value))
            continue
    return new_dict