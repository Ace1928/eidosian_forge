import re
import sys
from weakref import ReferenceType
import weakref
from debian._util import resolve_ref, _strI
from debian._deb822_repro._util import BufferingIterator
class Deb822Token:
    """A token is an atomic syntactical element from a deb822 file

    A file is parsed into a series of tokens.  If these tokens are converted to
    text in exactly the same order, you get exactly the same file - bit-for-bit.
    Accordingly ever bit of text in a file must be assigned to exactly one
    Deb822Token.
    """
    __slots__ = ('_text', '_parent_element', '__weakref__')

    def __init__(self, text):
        if text == '':
            raise ValueError('Tokens must have content')
        self._text = text
        self._parent_element = None
        self._verify_token_text()

    def __repr__(self):
        return "{clsname}('{text}')".format(clsname=self.__class__.__name__, text=self._text.replace('\n', '\\n'))

    def _verify_token_text(self):
        if '\n' in self._text:
            is_single_line_token = False
            if self.is_comment or isinstance(self, Deb822ErrorToken):
                is_single_line_token = True
            if not is_single_line_token and (not self.is_whitespace):
                raise ValueError('Only whitespace, error and comment tokens may contain newlines')
            if not self.text.endswith('\n'):
                raise ValueError('Tokens containing whitespace must end on a newline')
            if is_single_line_token and '\n' in self.text[:-1]:
                raise ValueError('Comments and error tokens must not contain embedded newlines (only end on one)')

    @property
    def is_whitespace(self):
        return False

    @property
    def is_comment(self):
        return False

    @property
    def text(self):
        return self._text

    def convert_to_text(self):
        return self._text

    @property
    def parent_element(self):
        return resolve_ref(self._parent_element)

    @parent_element.setter
    def parent_element(self, new_parent):
        self._parent_element = weakref.ref(new_parent) if new_parent is not None else None

    def clear_parent_if_parent(self, parent):
        if parent is self.parent_element:
            self._parent_element = None