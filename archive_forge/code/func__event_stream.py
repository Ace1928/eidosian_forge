import re
import sys
import warnings
from bs4.css import CSS
from bs4.formatter import (
def _event_stream(self, iterator=None):
    """Yield a sequence of events that can be used to reconstruct the DOM
        for this element.

        This lets us recreate the nested structure of this element
        (e.g. when formatting it as a string) without using recursive
        method calls.

        This is similar in concept to the SAX API, but it's a simpler
        interface designed for internal use. The events are different
        from SAX and the arguments associated with the events are Tags
        and other Beautiful Soup objects.

        :param iterator: An alternate iterator to use when traversing
         the tree.
        """
    tag_stack = []
    iterator = iterator or self.self_and_descendants
    for c in iterator:
        while tag_stack and c.parent != tag_stack[-1]:
            now_closed_tag = tag_stack.pop()
            yield (Tag.END_ELEMENT_EVENT, now_closed_tag)
        if isinstance(c, Tag):
            if c.is_empty_element:
                yield (Tag.EMPTY_ELEMENT_EVENT, c)
            else:
                yield (Tag.START_ELEMENT_EVENT, c)
                tag_stack.append(c)
                continue
        else:
            yield (Tag.STRING_ELEMENT_EVENT, c)
    while tag_stack:
        now_closed_tag = tag_stack.pop()
        yield (Tag.END_ELEMENT_EVENT, now_closed_tag)