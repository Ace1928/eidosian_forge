"""
**1.4 Logger (`logger.py`):**
- **Purpose:** Handles logging of all module operations.
- **Functions:**
  - `log(message, level)`: Logs a message at the specified level (INFO, ERROR, etc.).
"""


class Logger:
    """
    A meticulously designed logging class that provides extensive logging capabilities across various levels of severity.
    This class encapsulates advanced logging functionalities including file rotation, formatting customization, and systematic record-keeping,
    ensuring that all log entries are comprehensively recorded and easily traceable.

    Attributes:
        logger_instance (logging.Logger): The logger instance utilized for logging messages.
        log_file_directory (str): Directory path to the log file where logs are written.
        log_file_name (str): Name of the log file where logs are written.
        maximum_log_file_size (int): Maximum size in bytes before log rotation is triggered.
        retained_backup_logs (int): Number of backup log files to retain.
    """

    def __init__(
        self,
        logger_name: str = "AdvancedScriptSeparatorModuleLogger",
        log_directory: str = "logs",
        log_filename: str = "advanced_script_separator_module.log",
        maximum_log_size: int = 10485760,  # 10MB
        backup_count: int = 5,
    ):
        """
        Initializes the AdvancedLoggingSystem instance with a rotating file handler to manage log file size and backup, ensuring comprehensive
        and detailed logging of all operations within the module.

        Parameters:
            logger_name (str): Name of the logger, defaults to 'AdvancedScriptSeparatorModuleLogger'.
            log_directory (str): Directory where the log file is stored, defaults to 'logs'.
            log_filename (str): Name of the log file, defaults to 'advanced_script_separator_module.log'.
            maximum_log_size (int): Maximum size of the log file in bytes before rotation, defaults to 10MB.
            backup_count (int): Number of backup log files to maintain, defaults to 5.
        """
        # Ensure the directory for the log file exists
        full_log_path = os.path.join(log_directory, log_filename)
        os.makedirs(os.path.dirname(full_log_path), exist_ok=True)

        # Create and configure logger
        self.logger_instance = logging.getLogger(logger_name)
        self.logger_instance.setLevel(
            logging.DEBUG
        )  # Capture all types of log messages

        # Create a rotating file handler
        handler = RotatingFileHandler(
            full_log_path, maxBytes=maximum_log_size, backupCount=backup_count
        )

        # Define the log format with maximum detail
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

        # Add the handler to the logger
        self.logger_instance.addHandler(handler)

    def log_detailed_message(self, message: str, severity_level: str):
        """
        Logs a message at the specified logging level with utmost precision and detail, ensuring all relevant information is captured.

        Parameters:
            message (str): The message to log, detailed and specific to the context.
            severity_level (str): The severity level at which to log the message. Expected values include 'debug', 'info', 'warning', 'error', 'critical'.

        Raises:
            ValueError: If the logging level is not recognized, ensuring strict adherence to logging standards.
        """
        # Validate and convert the severity level to a valid logging method
        log_method = getattr(self.logger_instance, severity_level.lower(), None)
        if log_method is None:
            raise ValueError(
                f"Logging level '{severity_level}' is not valid. Use 'debug', 'info', 'warning', 'error', or 'critical'."
            )
        log_method(message)  # Log the message with detailed context and precision
