from __future__ import annotations
import os
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING
from babel.core import Locale
from babel.messages.catalog import Catalog, Message
from babel.util import _cmp, wraptext
def _add_message(self) -> None:
    """
        Add a message to the catalog based on the current parser state and
        clear the state ready to process the next message.
        """
    self.translations.sort()
    if len(self.messages) > 1:
        msgid = tuple((m.denormalize() for m in self.messages))
    else:
        msgid = self.messages[0].denormalize()
    if isinstance(msgid, (list, tuple)):
        string = ['' for _ in range(self.catalog.num_plurals)]
        for idx, translation in self.translations:
            if idx >= self.catalog.num_plurals:
                self._invalid_pofile('', self.offset, 'msg has more translations than num_plurals of catalog')
                continue
            string[idx] = translation.denormalize()
        string = tuple(string)
    else:
        string = self.translations[0][1].denormalize()
    msgctxt = self.context.denormalize() if self.context else None
    message = Message(msgid, string, list(self.locations), set(self.flags), self.auto_comments, self.user_comments, lineno=self.offset + 1, context=msgctxt)
    if self.obsolete:
        if not self.ignore_obsolete:
            self.catalog.obsolete[msgid] = message
    else:
        self.catalog[msgid] = message
    self.counter += 1
    self._reset_message_state()