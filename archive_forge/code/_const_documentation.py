from ._impl import Mismatch
Never match.

    That is::

        self.assertThat(x, Never())

    Will never match and always fail, no matter what ``x`` is. Included for
    completeness with :py:func:`.Always`, but if you find a use for this, let
    us know!
    