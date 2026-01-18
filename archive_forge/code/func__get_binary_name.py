import inspect
import logging
import logging.config
import logging.handlers
import os
def _get_binary_name():
    return os.path.basename(inspect.stack()[-1][1])