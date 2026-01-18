import email.feedparser
import email.header
import email.message
import email.parser
import email.policy
import sys
import typing
from typing import Dict, List, Optional, Tuple, Union, cast
def _parse_keywords(data: str) -> List[str]:
    """Split a string of comma-separate keyboards into a list of keywords."""
    return [k.strip() for k in data.split(',')]