import logging
from datetime import datetime
import os
class CustomLogger:

    def __init__(self, name, level=logging.DEBUG):
        """
        Initializes the custom logger with the specified name and level.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        if not os.path.exists('logs'):
            os.makedirs('logs')
        log_filename = datetime.now().strftime('logs/log_%Y-%m-%d_%H-%M-%S.log')
        fh = logging.FileHandler(log_filename)
        fh.setLevel(level)
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def get_logger(self):
        """
        Returns the logger instance.
        """
        return self.logger