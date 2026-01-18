from typing import Any
class StopDownload(Exception):
    """
    Stop the download of the body for a given response.
    The 'fail' boolean parameter indicates whether or not the resulting partial response
    should be handled by the request errback. Note that 'fail' is a keyword-only argument.
    """

    def __init__(self, *, fail: bool=True):
        super().__init__()
        self.fail = fail