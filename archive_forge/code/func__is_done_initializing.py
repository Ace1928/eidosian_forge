import logging
import time
import datetime
from concurrent import futures
import parlai.chat_service.utils.logging as log_utils
import parlai.chat_service.utils.misc as utils
def _is_done_initializing(fut):
    e = fut.exception()
    if e is not None:
        self._log('`module_initialize` returned with error {}'.format(repr(e)))
        if self.debug:
            raise e
    if fut.result():
        print(fut.result())
    if self.debug:
        print('DEBUG: Call to `module_initialize` has completed...')
    self.initialized = True