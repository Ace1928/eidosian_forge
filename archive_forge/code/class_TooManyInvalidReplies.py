import logging
import sys
from humanfriendly.compat import interactive_prompt
from humanfriendly.terminal import (
from humanfriendly.text import format, concatenate
class TooManyInvalidReplies(Exception):
    """Raised by interactive prompts when they've received too many invalid inputs."""