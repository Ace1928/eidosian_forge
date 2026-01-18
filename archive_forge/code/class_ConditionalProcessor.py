from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from six.moves import range
from prompt_toolkit.cache import SimpleCache
from prompt_toolkit.document import Document
from prompt_toolkit.enums import SEARCH_BUFFER
from prompt_toolkit.filters import to_cli_filter, ViInsertMultipleMode
from prompt_toolkit.layout.utils import token_list_to_text
from prompt_toolkit.reactive import Integer
from prompt_toolkit.token import Token
from .utils import token_list_len, explode_tokens
import re
class ConditionalProcessor(Processor):
    """
    Processor that applies another processor, according to a certain condition.
    Example::

        # Create a function that returns whether or not the processor should
        # currently be applied.
        def highlight_enabled(cli):
            return true_or_false

        # Wrapt it in a `ConditionalProcessor` for usage in a `BufferControl`.
        BufferControl(input_processors=[
            ConditionalProcessor(HighlightSearchProcessor(),
                                 Condition(highlight_enabled))])

    :param processor: :class:`.Processor` instance.
    :param filter: :class:`~prompt_toolkit.filters.CLIFilter` instance.
    """

    def __init__(self, processor, filter):
        assert isinstance(processor, Processor)
        self.processor = processor
        self.filter = to_cli_filter(filter)

    def apply_transformation(self, cli, document, lineno, source_to_display, tokens):
        if self.filter(cli):
            return self.processor.apply_transformation(cli, document, lineno, source_to_display, tokens)
        else:
            return Transformation(tokens)

    def has_focus(self, cli):
        if self.filter(cli):
            return self.processor.has_focus(cli)
        else:
            return False

    def __repr__(self):
        return '%s(processor=%r, filter=%r)' % (self.__class__.__name__, self.processor, self.filter)