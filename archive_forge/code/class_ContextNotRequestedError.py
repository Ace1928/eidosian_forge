from oslo_utils.excutils import CausedByException
from oslo_db._i18n import _
class ContextNotRequestedError(AttributeError):
    """Error raised when requesting a not-setup enginefacade attribute.

    This applies to the ``session`` and ``connection`` attributes
    of a user-defined context and/or RequestContext object, when they
    are accessed *within* the scope of an enginefacade decorator
    or context manager, but the context has not requested that
    attribute (e.g. like "with enginefacade.connection.using(context)"
    and "context.session" is requested).

    """