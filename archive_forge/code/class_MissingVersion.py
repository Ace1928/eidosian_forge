import logging
import re
class MissingVersion(Exception):
    """The version= line is missing."""