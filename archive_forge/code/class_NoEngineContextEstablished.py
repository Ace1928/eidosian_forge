from oslo_utils.excutils import CausedByException
from oslo_db._i18n import _
class NoEngineContextEstablished(AttributeError):
    """Error raised for enginefacade attribute access with no context.


    This applies to the ``session`` and ``connection`` attributes
    of a user-defined context and/or RequestContext object, when they
    are accessed *outside* of the scope of an enginefacade decorator
    or context manager.

    The exception is a subclass of AttributeError so that
    normal Python missing attribute behaviors are maintained, such
    as support for ``getattr(context, 'session', None)``.


    """