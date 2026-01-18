import copy
import logging
from s3transfer.utils import get_callbacks
def _execute_main(self, kwargs):
    params_to_exclude = ['data']
    kwargs_to_display = self._get_kwargs_with_params_to_exclude(kwargs, params_to_exclude)
    logger.debug('Executing task %s with kwargs %s' % (self, kwargs_to_display))
    return_value = self._main(**kwargs)
    if self._is_final:
        self._transfer_coordinator.set_result(return_value)
    return return_value