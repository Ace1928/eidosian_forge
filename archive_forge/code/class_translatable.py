from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import docutils
from docutils import nodes
from docutils.nodes import Element
class translatable(nodes.Node):
    """Node which supports translation.

    The translation goes forward with following steps:

    1. Preserve original translatable messages
    2. Apply translated messages from message catalog
    3. Extract preserved messages (for gettext builder)

    The translatable nodes MUST preserve original messages.
    And these messages should not be overridden at applying step.
    Because they are used at final step; extraction.
    """

    def preserve_original_messages(self) -> None:
        """Preserve original translatable messages."""
        raise NotImplementedError

    def apply_translated_message(self, original_message: str, translated_message: str) -> None:
        """Apply translated message."""
        raise NotImplementedError

    def extract_original_messages(self) -> Sequence[str]:
        """Extract translation messages.

        :returns: list of extracted messages or messages generator
        """
        raise NotImplementedError