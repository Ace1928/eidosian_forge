from __future__ import annotations
import os
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING
from babel.core import Locale
from babel.messages.catalog import Catalog, Message
from babel.util import _cmp, wraptext
def _sort_messages(messages: Iterable[Message], sort_by: Literal['message', 'location']) -> list[Message]:
    """
    Sort the given message iterable by the given criteria.

    Always returns a list.

    :param messages: An iterable of Messages.
    :param sort_by: Sort by which criteria? Options are `message` and `location`.
    :return: list[Message]
    """
    messages = list(messages)
    if sort_by == 'message':
        messages.sort()
    elif sort_by == 'location':
        messages.sort(key=lambda m: m.locations)
    return messages