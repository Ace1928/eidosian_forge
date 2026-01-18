import re
import ast
import logging
from typing import List, Dict, Any, Union
import numpy as np
import logging
from typing import List
import os
import logging
from typing import Dict, List, Union
import os
import logging
from logging.handlers import RotatingFileHandler
from typing import Union
import json
import xml.etree.ElementTree as ET
import logging
import os
import subprocess
import logging
from typing import List
import ast
import logging
from typing import List, Dict
import logging
from typing import Dict
import ast
import networkx as nx
import matplotlib.pyplot as plt
from typing import List
import logging
from typing import Type, Union
class PreciseLogger:
    """
    **1.4 Logger (`logger.py`):**
    - **Purpose:** Manages the logging of all module operations with precision and detail.
    - **Functions:**
    - `log_message_with_detailed_context(message: str, level: str) -> None`: Logs a message at the specified level (DEBUG, INFO, WARNING, ERROR, CRITICAL) with comprehensive details.
    """
    '\n    A comprehensive logging system meticulously designed to handle logs across various severity levels with high precision and detail. This class incorporates file rotation,\n    custom formatting, and systematic record-keeping to ensure that all log entries are meticulously recorded and easily traceable.\n\n    Attributes:\n        logger (logging.Logger): The logger instance used for logging messages.\n        log_file_path (str): Full path to the log file where logs are stored.\n        max_log_size_bytes (int): Maximum size in bytes before log rotation is triggered.\n        backup_logs_count (int): Number of backup log files to retain.\n    '

    def __init__(self, logger_name: str='PreciseScriptLogger', log_directory: str='logs', log_filename: str='precise_script.log', max_log_size: int=10485760, backup_count: int=5) -> None:
        """
        Initializes the PreciseLogger with a rotating file handler to manage log file size and backup, ensuring detailed and comprehensive logging.

        Parameters:
            logger_name (str): Name of the logger, defaults to 'PreciseScriptLogger'.
            log_directory (str): Directory where the log file is stored, defaults to 'logs'.
            log_filename (str): Name of the log file, defaults to 'precise_script.log'.
            max_log_size (int): Maximum size of the log file in bytes before rotation, defaults to 10MB.
            backup_count (int): Number of backup log files to maintain, defaults to 5.
        """
        self.log_file_path = os.path.join(log_directory, log_filename)
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        handler = RotatingFileHandler(self.log_file_path, maxBytes=max_log_size, backupCount=backup_count)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def log_message_with_detailed_context(self, message: str, severity_level: str) -> None:
        """
        Logs a message at the specified logging level with utmost precision and detail, ensuring all relevant information is captured.

        Parameters:
            message (str): The message to log, detailed and specific to the context.
            severity_level (str): The severity level at which to log the message. Expected values include 'debug', 'info', 'warning', 'error', 'critical'.

        Raises:
            ValueError: If the logging level is not recognized, ensuring strict adherence to logging standards.
        """
        log_method = getattr(self.logger, severity_level.lower(), None)
        if log_method is None:
            raise ValueError(f"Logging level '{severity_level}' is not valid. Use 'debug', 'info', 'warning', 'error', or 'critical'.")
        log_method(message)