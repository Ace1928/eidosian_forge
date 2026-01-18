from __future__ import annotations
import asyncio
import math
from typing import TYPE_CHECKING
from prompt_toolkit.application.run_in_terminal import in_terminal
from prompt_toolkit.completion import (
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding.key_bindings import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.utils import get_cwidth
def _create_more_session(message: str='--MORE--') -> PromptSession[bool]:
    """
    Create a `PromptSession` object for displaying the "--MORE--".
    """
    from prompt_toolkit.shortcuts import PromptSession
    bindings = KeyBindings()

    @bindings.add(' ')
    @bindings.add('y')
    @bindings.add('Y')
    @bindings.add(Keys.ControlJ)
    @bindings.add(Keys.ControlM)
    @bindings.add(Keys.ControlI)
    def _yes(event: E) -> None:
        event.app.exit(result=True)

    @bindings.add('n')
    @bindings.add('N')
    @bindings.add('q')
    @bindings.add('Q')
    @bindings.add(Keys.ControlC)
    def _no(event: E) -> None:
        event.app.exit(result=False)

    @bindings.add(Keys.Any)
    def _ignore(event: E) -> None:
        """Disable inserting of text."""
    return PromptSession(message, key_bindings=bindings, erase_when_done=True)