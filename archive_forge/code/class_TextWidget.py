from abc import ABCMeta, abstractmethod
from tkinter import (
from tkinter.filedialog import asksaveasfilename
from nltk.util import in_idle
class TextWidget(CanvasWidget):
    """
    A canvas widget that displays a single string of text.

    Attributes:
      - ``color``: the color of the text.
      - ``font``: the font used to display the text.
      - ``justify``: justification for multi-line texts.  Valid values
        are ``left``, ``center``, and ``right``.
      - ``width``: the width of the text.  If the text is wider than
        this width, it will be line-wrapped at whitespace.
      - ``draggable``: whether the text can be dragged by the user.
    """

    def __init__(self, canvas, text, **attribs):
        """
        Create a new text widget.

        :type canvas: Tkinter.Canvas
        :param canvas: This canvas widget's canvas.
        :type text: str
        :param text: The string of text to display.
        :param attribs: The new canvas widget's attributes.
        """
        self._text = text
        self._tag = canvas.create_text(1, 1, text=text)
        CanvasWidget.__init__(self, canvas, **attribs)

    def __setitem__(self, attr, value):
        if attr in ('color', 'font', 'justify', 'width'):
            if attr == 'color':
                attr = 'fill'
            self.canvas().itemconfig(self._tag, {attr: value})
        else:
            CanvasWidget.__setitem__(self, attr, value)

    def __getitem__(self, attr):
        if attr == 'width':
            return int(self.canvas().itemcget(self._tag, attr))
        elif attr in ('color', 'font', 'justify'):
            if attr == 'color':
                attr = 'fill'
            return self.canvas().itemcget(self._tag, attr)
        else:
            return CanvasWidget.__getitem__(self, attr)

    def _tags(self):
        return [self._tag]

    def text(self):
        """
        :return: The text displayed by this text widget.
        :rtype: str
        """
        return self.canvas().itemcget(self._tag, 'TEXT')

    def set_text(self, text):
        """
        Change the text that is displayed by this text widget.

        :type text: str
        :param text: The string of text to display.
        :rtype: None
        """
        self.canvas().itemconfig(self._tag, text=text)
        if self.parent() is not None:
            self.parent().update(self)

    def __repr__(self):
        return '[Text: %r]' % self._text