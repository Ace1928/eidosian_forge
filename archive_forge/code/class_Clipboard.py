from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
import six
from prompt_toolkit.selection import SelectionType
class Clipboard(with_metaclass(ABCMeta, object)):
    """
    Abstract baseclass for clipboards.
    (An implementation can be in memory, it can share the X11 or Windows
    keyboard, or can be persistent.)
    """

    @abstractmethod
    def set_data(self, data):
        """
        Set data to the clipboard.

        :param data: :class:`~.ClipboardData` instance.
        """

    def set_text(self, text):
        """
        Shortcut for setting plain text on clipboard.
        """
        assert isinstance(text, six.string_types)
        self.set_data(ClipboardData(text))

    def rotate(self):
        """
        For Emacs mode, rotate the kill ring.
        """

    @abstractmethod
    def get_data(self):
        """
        Return clipboard data.
        """