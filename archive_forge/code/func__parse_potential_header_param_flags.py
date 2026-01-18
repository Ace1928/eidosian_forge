import json
import urllib.parse
from tensorboard import context
from tensorboard import errors
def _parse_potential_header_param_flags(self, header_string):
    if not header_string:
        return {}
    try:
        header_feature_flags = json.loads(header_string)
    except json.JSONDecodeError:
        raise errors.InvalidArgumentError('X-TensorBoard-Feature-Flags cannot be JSON decoded.')
    if not isinstance(header_feature_flags, dict):
        raise errors.InvalidArgumentError('X-TensorBoard-Feature-Flags cannot be decoded to a dict.')
    return header_feature_flags