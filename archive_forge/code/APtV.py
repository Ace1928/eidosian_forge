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

        # Create logs directory if it doesn't exist
        if not os.path.exists("logs"):
            os.makedirs("logs")

        # Define file name format for the log files
        log_filename = datetime.now().strftime("logs/log_%Y-%m-%d_%H-%M-%S.log")

        # Create file handler which logs even debug messages
        fh = logging.FileHandler(log_filename)
        fh.setLevel(level)

        # Create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)

        # Create formatter and add it to the handlers
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add the handlers to the logger
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def get_logger(self):
        """
        Returns the logger instance.
        """
        return self.logger


# Example usage
if __name__ == "__main__":
    custom_logger = CustomLogger("fractal_neural_network").get_logger()
    custom_logger.debug("This is a debug message")
    custom_logger.info("This is an info message")
    custom_logger.warning("This is a warning message")
    custom_logger.error("This is an error message")
    custom_logger.critical("This is a critical message")
