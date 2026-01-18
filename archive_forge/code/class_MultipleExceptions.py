from oslo_utils import excutils
from neutron_lib._i18n import _
class MultipleExceptions(Exception):
    """Container for multiple exceptions encountered.

    The API layer of Neutron will automatically unpack, translate,
    filter, and combine the inner exceptions in any exception derived
    from this class.
    """

    def __init__(self, exceptions, *args, **kwargs):
        """Create a new instance wrapping the exceptions.

        :param exceptions: The inner exceptions this instance is composed of.
        :param args: Passed onto parent constructor.
        :param kwargs: Passed onto parent constructor.
        """
        super().__init__(*args, **kwargs)
        self.inner_exceptions = exceptions