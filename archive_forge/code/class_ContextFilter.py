import logging
import re
from logging.config import dictConfig
import threading
from typing import Union
class ContextFilter(logging.Filter):
    """A filter that adds ray context info to log records.

    This filter adds a package name to append to the message as well as information
    about what worker emitted the message, if applicable.
    """
    logger_regex = re.compile('ray(\\.(?P<subpackage>\\w+))?(\\..*)?')
    package_message_names = {'air': 'AIR', 'data': 'Data', 'rllib': 'RLlib', 'serve': 'Serve', 'train': 'Train', 'tune': 'Tune', 'workflow': 'Workflow'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def filter(self, record: logging.LogRecord) -> bool:
        """Add context information to the log record.

        This filter adds a package name from where the message was generated as
        well as the worker IP address, if applicable.

        Args:
            record: Record to be filtered

        Returns:
            True if the record is to be logged, False otherwise. (This filter only
            adds context, so records are always logged.)
        """
        match = self.logger_regex.search(record.name)
        if match and match['subpackage'] in self.package_message_names:
            record.package = f'[Ray {self.package_message_names[match['subpackage']]}]'
        else:
            record.package = ''
        return True