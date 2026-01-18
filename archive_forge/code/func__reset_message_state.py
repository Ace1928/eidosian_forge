from __future__ import annotations
import os
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING
from babel.core import Locale
from babel.messages.catalog import Catalog, Message
from babel.util import _cmp, wraptext
def _reset_message_state(self) -> None:
    self.messages = []
    self.translations = []
    self.locations = []
    self.flags = []
    self.user_comments = []
    self.auto_comments = []
    self.context = None
    self.obsolete = False
    self.in_msgid = False
    self.in_msgstr = False
    self.in_msgctxt = False