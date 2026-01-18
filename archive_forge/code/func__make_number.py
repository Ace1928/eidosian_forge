import re
def _make_number(self, ls, title, nc, s1):
    """Format cutting position information as a string (PRIVATE).

        Returns a string in the form::

            title.

            enzyme which cut 1 time:

            enzyme1     :   position1.

            enzyme which cut 2 times:

            enzyme2     :   position1, position2.
            ...

        Arguments:
         - ls is a list of cutting enzymes.
         - title is the title.
         - nc is a list of non cutting enzymes.
         - s1 is the sentence before the non cutting enzymes.
        """
    return self._make_number_only(ls, title) + self._make_nocut_only(nc, s1)