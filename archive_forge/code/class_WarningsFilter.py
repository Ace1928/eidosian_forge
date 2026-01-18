import warnings
import fixtures
class WarningsFilter(fixtures.Fixture):
    """Configure warnings filters.

    While ``WarningsFilter`` is active, warnings will be filtered per
    configuration.
    """

    def __init__(self, filters=None):
        """Create a WarningsFilter fixture.

        :param filters: An optional list of dictionaries with arguments
            corresponding to the arguments to
            :py:func:`warnings.filterwarnings`. For example::

                [
                    {
                        'action': 'ignore',
                        'message': 'foo',
                        'category': DeprecationWarning,
                    },
                ]

            Order is important: entries closer to the front of the list
            override entries later in the list, if both match a particular
            warning.

            Alternatively, you can configure warnings within the context of the
            fixture.

            See `the Python documentation`__ for more information.

        __: https://docs.python.org/3/library/warnings.html#the-warnings-filter
        """
        super().__init__()
        self.filters = filters or []

    def _setUp(self):
        self._original_warning_filters = warnings.filters[:]
        for filt in self.filters:
            warnings.filterwarnings(**filt)
        self.addCleanup(self._reset_warning_filters)

    def _reset_warning_filters(self):
        warnings.filters[:] = self._original_warning_filters