from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
import six
from prompt_toolkit.selection import SelectionType
class ClipboardData(object):
    """
    Text on the clipboard.

    :param text: string
    :param type: :class:`~prompt_toolkit.selection.SelectionType`
    """

    def __init__(self, text='', type=SelectionType.CHARACTERS):
        assert isinstance(text, six.string_types)
        assert type in (SelectionType.CHARACTERS, SelectionType.LINES, SelectionType.BLOCK)
        self.text = text
        self.type = type