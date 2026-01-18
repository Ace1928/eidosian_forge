from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
class CompleteEvent(object):
    """
    Event that called the completer.

    :param text_inserted: When True, it means that completions are requested
        because of a text insert. (`Buffer.complete_while_typing`.)
    :param completion_requested: When True, it means that the user explicitely
        pressed the `Tab` key in order to view the completions.

    These two flags can be used for instance to implemented a completer that
    shows some completions when ``Tab`` has been pressed, but not
    automatically when the user presses a space. (Because of
    `complete_while_typing`.)
    """

    def __init__(self, text_inserted=False, completion_requested=False):
        assert not (text_inserted and completion_requested)
        self.text_inserted = text_inserted
        self.completion_requested = completion_requested

    def __repr__(self):
        return '%s(text_inserted=%r, completion_requested=%r)' % (self.__class__.__name__, self.text_inserted, self.completion_requested)