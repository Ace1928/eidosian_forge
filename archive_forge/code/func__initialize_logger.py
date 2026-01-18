import logging
import os
import ray
from ray._private.ray_constants import LOGGER_FORMAT, LOGGER_LEVEL
def _initialize_logger(self) -> logging.Logger:
    """Internal method to initialize the logger and the extra file handler
        for writing to the Dataset log file. Not intended (nor necessary)
        to call explicitly. Assumes that `ray.init()` has already been called prior
        to calling this method; otherwise raises a `ValueError`."""
    stdout_logger = logging.getLogger(self.log_name)
    stdout_logger.setLevel(LOGGER_LEVEL.upper())
    logger = logging.getLogger(f'{self.log_name}.logfile')
    logger.setLevel(LOGGER_LEVEL.upper())
    global_node = ray._private.worker._global_node
    if global_node is not None:
        session_dir = global_node.get_session_dir_path()
        datasets_log_path = os.path.join(session_dir, DatasetLogger.DEFAULT_DATASET_LOG_PATH)
        file_log_formatter = logging.Formatter(fmt=LOGGER_FORMAT)
        file_log_handler = logging.FileHandler(datasets_log_path)
        file_log_handler.setLevel(LOGGER_LEVEL.upper())
        file_log_handler.setFormatter(file_log_formatter)
        logger.addHandler(file_log_handler)
    return logger