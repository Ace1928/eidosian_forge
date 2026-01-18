from __future__ import unicode_literals
class SelectionState(object):
    """
    State of the current selection.

    :param original_cursor_position: int
    :param type: :class:`~.SelectionType`
    """

    def __init__(self, original_cursor_position=0, type=SelectionType.CHARACTERS):
        self.original_cursor_position = original_cursor_position
        self.type = type

    def __repr__(self):
        return '%s(original_cursor_position=%r, type=%r)' % (self.__class__.__name__, self.original_cursor_position, self.type)