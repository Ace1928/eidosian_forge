import sys
import logging
class ParlaiLogger(logging.Logger):

    def __init__(self, name, console_level=INFO):
        """
        Initialize the logger object.

        :param name:
            Name of the logger
        :param console_level:
            minimum level of messages logged to console
        """
        super().__init__(name, console_level)
        self.streamHandler = logging.StreamHandler(sys.stdout)
        self.prefix = None
        self.streamHandler.setFormatter(self._build_formatter())
        super().addHandler(self.streamHandler)

    def _build_formatter(self):
        prefix_format = f'{self.prefix} ' if self.prefix else ''
        if COLORED_LOGS and sys.stdout.isatty():
            return coloredlogs.ColoredFormatter(prefix_format + COLORED_FORMAT, datefmt=CONSOLE_DATE_FORMAT, level_styles=COLORED_LEVEL_STYLES, field_styles={})
        elif sys.stdout.isatty():
            return logging.Formatter(prefix_format + CONSOLE_FORMAT, datefmt=CONSOLE_DATE_FORMAT)
        else:
            return logging.Formatter(prefix_format + LOGFILE_FORMAT, datefmt=LOGFILE_DATE_FORMAT)

    def log(self, msg, level=INFO):
        """
        Default Logging function.
        """
        super().log(level, msg)

    def add_format_prefix(self, prefix):
        """
        Include `prefix` in all future logging statements.
        """
        self.prefix = prefix
        self.streamHandler.setFormatter(self._build_formatter())

    def mute(self):
        """
        Stop logging to stdout.
        """
        self.prev_level = self.streamHandler.level
        self.streamHandler.level = ERROR
        return self.prev_level

    def unmute(self):
        """
        Resume logging to stdout.
        """
        self.streamHandler.level = self.prev_level