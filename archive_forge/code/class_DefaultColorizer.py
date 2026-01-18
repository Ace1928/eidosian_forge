import logging
from logutils.colorize import ColorizingStreamHandler
class DefaultColorizer(ColorizingStreamHandler):
    level_map = {logging.DEBUG: (None, 'blue', True), logging.INFO: (None, None, True), logging.WARNING: (None, 'yellow', True), logging.ERROR: (None, 'red', True), logging.CRITICAL: (None, 'red', True)}