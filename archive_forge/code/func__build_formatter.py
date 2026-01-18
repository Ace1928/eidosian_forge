import sys
import logging
def _build_formatter(self):
    prefix_format = f'{self.prefix} ' if self.prefix else ''
    if COLORED_LOGS and sys.stdout.isatty():
        return coloredlogs.ColoredFormatter(prefix_format + COLORED_FORMAT, datefmt=CONSOLE_DATE_FORMAT, level_styles=COLORED_LEVEL_STYLES, field_styles={})
    elif sys.stdout.isatty():
        return logging.Formatter(prefix_format + CONSOLE_FORMAT, datefmt=CONSOLE_DATE_FORMAT)
    else:
        return logging.Formatter(prefix_format + LOGFILE_FORMAT, datefmt=LOGFILE_DATE_FORMAT)