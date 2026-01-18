from __future__ import annotations
import re
import sys
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING, Optional
from pyglet import event
from pyglet.text import runlist
class AbstractDocument(event.EventDispatcher):
    """Abstract document interface used by all :py:mod:`pyglet.text` classes.

    This class can be overridden to interface pyglet with a third-party
    document format.  It may be easier to implement the document format in
    terms of one of the supplied concrete classes :py:class:`~pyglet.text.document.FormattedDocument` or
    :py:class:`~pyglet.text.document.UnformattedDocument`.
    """
    _previous_paragraph_re = re.compile(u'\n[^\n\u2029]*$')
    _next_paragraph_re = re.compile(u'[\n\u2029]')

    def __init__(self, text=''):
        super().__init__()
        self._text = u''
        self._elements = []
        if text:
            self.insert_text(0, text)

    @property
    def text(self):
        """Document text.

        For efficient incremental updates, use the :py:func:`~pyglet.text.document.AbstractDocument.insert_text` and
        :py:func:`~pyglet.text.document.AbstractDocument.delete_text` methods instead of replacing this property.

        :type: str
        """
        return self._text

    @text.setter
    def text(self, text):
        if text == self._text:
            return
        self.delete_text(0, len(self._text))
        self.insert_text(0, text)

    def get_paragraph_start(self, pos):
        """Get the starting position of a paragraph.

        :Parameters:
            `pos` : int
                Character position within paragraph.

        :rtype: int
        """
        if self._text[:pos + 1].endswith('\n') or self._text[:pos + 1].endswith(u'\u2029'):
            return pos
        m = self._previous_paragraph_re.search(self._text, 0, pos + 1)
        if not m:
            return 0
        return m.start() + 1

    def get_paragraph_end(self, pos):
        """Get the end position of a paragraph.

        :Parameters:
            `pos` : int
                Character position within paragraph.

        :rtype: int
        """
        m = self._next_paragraph_re.search(self._text, pos)
        if not m:
            return len(self._text)
        return m.start() + 1

    def get_style_runs(self, attribute):
        """Get a style iterator over the given style attribute.

        :Parameters:
            `attribute` : str
                Name of style attribute to query.

        :rtype: `AbstractRunIterator`
        """
        raise NotImplementedError('abstract')

    def get_style(self, attribute, position=0):
        """Get an attribute style at the given position.

        :Parameters:
            `attribute` : str
                Name of style attribute to query.
            `position` : int
                Character position of document to query.

        :return: The style set for the attribute at the given position.
        """
        raise NotImplementedError('abstract')

    def get_style_range(self, attribute, start, end):
        """Get an attribute style over the given range.

        If the style varies over the range, `STYLE_INDETERMINATE` is returned.

        :Parameters:
            `attribute` : str
                Name of style attribute to query.
            `start` : int
                Starting character position.
            `end` : int
                Ending character position (exclusive).

        :return: The style set for the attribute over the given range, or
            `STYLE_INDETERMINATE` if more than one value is set.
        """
        iterable = self.get_style_runs(attribute)
        _, value_end, value = next(iterable.ranges(start, end))
        if value_end < end:
            return STYLE_INDETERMINATE
        else:
            return value

    def get_font_runs(self, dpi=None):
        """Get a style iterator over the `pyglet.font.Font` instances used in
        the document.

        The font instances are created on-demand by inspection of the
        ``font_name``, ``font_size``, ``bold`` and ``italic`` style
        attributes.

        :Parameters:
            `dpi` : float
                Optional resolution to construct fonts at.  See
                :py:func:`pyglet.font.load`.

        :rtype: `AbstractRunIterator`
        """
        raise NotImplementedError('abstract')

    def get_font(self, position, dpi=None):
        """Get the font instance used at the given position.

        :see: `get_font_runs`

        :Parameters:
            `position` : int
                Character position of document to query.
            `dpi` : float
                Optional resolution to construct fonts at.  See
                :py:func:`pyglet.font.load`.

        :rtype: `pyglet.font.Font`
        :return: The font at the given position.
        """
        raise NotImplementedError('abstract')

    def insert_text(self, start, text, attributes=None):
        """Insert text into the document.

        :Parameters:
            `start` : int
                Character insertion point within document.
            `text` : str
                Text to insert.
            `attributes` : dict
                Optional dictionary giving named style attributes of the
                inserted text.

        """
        self._insert_text(start, text, attributes)
        self.dispatch_event('on_insert_text', start, text)

    def _insert_text(self, start, text, attributes):
        self._text = u''.join((self._text[:start], text, self._text[start:]))
        len_text = len(text)
        for element in self._elements:
            if element._position >= start:
                element._position += len_text

    def delete_text(self, start, end):
        """Delete text from the document.

        :Parameters:
            `start` : int
                Starting character position to delete from.
            `end` : int
                Ending character position to delete to (exclusive).

        """
        self._delete_text(start, end)
        self.dispatch_event('on_delete_text', start, end)

    def _delete_text(self, start, end):
        for element in list(self._elements):
            if start <= element._position < end:
                self._elements.remove(element)
            elif element._position >= end:
                element._position -= end - start
        self._text = self._text[:start] + self._text[end:]

    def insert_element(self, position, element, attributes=None):
        """Insert a element into the document.

        See the :py:class:`~pyglet.text.document.InlineElement` class
        documentation for details of usage.

        :Parameters:
            `position` : int
                Character insertion point within document.
            `element` : `~pyglet.text.document.InlineElement`
                Element to insert.
            `attributes` : dict
                Optional dictionary giving named style attributes of the
                inserted text.

        """
        assert element._position is None, 'Element is already in a document.'
        self.insert_text(position, '\x00', attributes)
        element._position = position
        self._elements.append(element)
        self._elements.sort(key=lambda d: d.position)

    def get_element(self, position):
        """Get the element at a specified position.

        :Parameters:
            `position` : int
                Position in the document of the element.

        :rtype: :py:class:`~pyglet.text.document.InlineElement`
        """
        for element in self._elements:
            if element._position == position:
                return element
        raise RuntimeError(f'No element at position {position}')

    def set_style(self, start, end, attributes):
        """Set text style of some or all of the document.

        :Parameters:
            `start` : int
                Starting character position.
            `end` : int
                Ending character position (exclusive).
            `attributes` : dict
                Dictionary giving named style attributes of the text.

        """
        self._set_style(start, end, attributes)
        self.dispatch_event('on_style_text', start, end, attributes)

    def _set_style(self, start, end, attributes):
        raise NotImplementedError('abstract')

    def set_paragraph_style(self, start, end, attributes):
        """Set the style for a range of paragraphs.

        This is a convenience method for `set_style` that aligns the
        character range to the enclosing paragraph(s).

        :Parameters:
            `start` : int
                Starting character position.
            `end` : int
                Ending character position (exclusive).
            `attributes` : dict
                Dictionary giving named style attributes of the paragraphs.

        """
        start = self.get_paragraph_start(start)
        end = self.get_paragraph_end(end)
        self._set_style(start, end, attributes)
        self.dispatch_event('on_style_text', start, end, attributes)
    if _is_pyglet_doc_run:

        def on_insert_text(self, start, text):
            """Text was inserted into the document.

            :Parameters:
                `start` : int
                    Character insertion point within document.
                `text` : str
                    The text that was inserted.

            :event:
            """

        def on_delete_text(self, start, end):
            """Text was deleted from the document.

            :Parameters:
                `start` : int
                    Starting character position of deleted text.
                `end` : int
                    Ending character position of deleted text (exclusive).

            :event:
            """

        def on_style_text(self, start, end, attributes):
            """Text character style was modified.

            :Parameters:
                `start` : int
                    Starting character position of modified text.
                `end` : int
                    Ending character position of modified text (exclusive).
                `attributes` : dict
                    Dictionary giving updated named style attributes of the
                    text.

            :event:
            """