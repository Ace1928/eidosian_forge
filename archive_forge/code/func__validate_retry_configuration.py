import copy
from botocore.compat import OrderedDict
from botocore.endpoint import DEFAULT_TIMEOUT, MAX_POOL_CONNECTIONS
from botocore.exceptions import (
def _validate_retry_configuration(self, retries):
    valid_options = ('max_attempts', 'mode', 'total_max_attempts')
    valid_modes = ('legacy', 'standard', 'adaptive')
    if retries is not None:
        for key, value in retries.items():
            if key not in valid_options:
                raise InvalidRetryConfigurationError(retry_config_option=key, valid_options=valid_options)
            if key == 'max_attempts' and value < 0:
                raise InvalidMaxRetryAttemptsError(provided_max_attempts=value, min_value=0)
            if key == 'total_max_attempts' and value < 1:
                raise InvalidMaxRetryAttemptsError(provided_max_attempts=value, min_value=1)
            if key == 'mode' and value not in valid_modes:
                raise InvalidRetryModeError(provided_retry_mode=value, valid_modes=valid_modes)