import contextlib
import logging
import logging.config
import re
import sys
def _configure_mlflow_loggers(root_module_name):
    logging.config.dictConfig({'version': 1, 'disable_existing_loggers': False, 'formatters': {'mlflow_formatter': {'format': LOGGING_LINE_FORMAT, 'datefmt': LOGGING_DATETIME_FORMAT}}, 'handlers': {'mlflow_handler': {'formatter': 'mlflow_formatter', 'class': 'logging.StreamHandler', 'stream': MLFLOW_LOGGING_STREAM}}, 'loggers': {root_module_name: {'handlers': ['mlflow_handler'], 'level': 'INFO', 'propagate': False}}})