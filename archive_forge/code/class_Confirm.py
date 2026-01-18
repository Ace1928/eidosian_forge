from typing import Any, Generic, List, Optional, TextIO, TypeVar, Union, overload
from . import get_console
from .console import Console
from .text import Text, TextType
class Confirm(PromptBase[bool]):
    """A yes / no confirmation prompt.

    Example:
        >>> if Confirm.ask("Continue"):
                run_job()

    """
    response_type = bool
    validate_error_message = '[prompt.invalid]Please enter Y or N'
    choices: List[str] = ['y', 'n']

    def render_default(self, default: DefaultType) -> Text:
        """Render the default as (y) or (n) rather than True/False."""
        yes, no = self.choices
        return Text(f'({yes})' if default else f'({no})', style='prompt.default')

    def process_response(self, value: str) -> bool:
        """Convert choices to a bool."""
        value = value.strip().lower()
        if value not in self.choices:
            raise InvalidResponse(self.validate_error_message)
        return value == self.choices[0]