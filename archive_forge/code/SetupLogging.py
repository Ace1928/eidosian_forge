
import logging

def setup_logging(log_file='application.log', level=logging.INFO):
    """
    Sets up the logging for the application.

    Args:
    log_file (str): The name of the log file.
    level (logging.Level): The logging level (e.g., INFO, ERROR, DEBUG).
    """
    logging.basicConfig(filename=log_file, level=level,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Adding an example log message
    logging.info('Logging setup complete.')

# Example usage
if __name__ == "__main__":
    setup_logging(level=logging.DEBUG)
    logging.debug('This is a debug message.')
