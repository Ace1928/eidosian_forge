from __future__ import unicode_literals
import functools
import re
from datetime import timedelta
import logging
import io
def _should_skip_sub(subtitle):
    """
    Check if a subtitle should be skipped based on the rules in
    SUBTITLE_SKIP_CONDITIONS.

    :param subtitle: A :py:class:`Subtitle` to check whether to skip
    :raises _ShouldSkipException: If the subtitle should be skipped
    """
    for info_msg, sub_skipper in SUBTITLE_SKIP_CONDITIONS:
        if sub_skipper(subtitle):
            raise _ShouldSkipException(info_msg)