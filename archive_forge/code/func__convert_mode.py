from .. import errors
from ..constants import IS_WINDOWS_PLATFORM
from ..utils import (
@staticmethod
def _convert_mode(original_mode):
    if original_mode == 'global-job':
        return 'GlobalJob'
    if original_mode == 'replicated-job':
        return 'ReplicatedJob'
    return original_mode