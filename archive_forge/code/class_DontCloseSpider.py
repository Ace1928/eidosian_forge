from typing import Any
class DontCloseSpider(Exception):
    """Request the spider not to be closed yet"""
    pass