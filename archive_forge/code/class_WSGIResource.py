from collections.abc import Sequence
from sys import exc_info
from warnings import warn
from zope.interface import implementer
from twisted.internet.threads import blockingCallFromThread
from twisted.logger import Logger
from twisted.python.failure import Failure
from twisted.web.http import INTERNAL_SERVER_ERROR
from twisted.web.resource import IResource
from twisted.web.server import NOT_DONE_YET
@implementer(IResource)
class WSGIResource:
    """
    An L{IResource} implementation which delegates responsibility for all
    resources hierarchically inferior to it to a WSGI application.

    @ivar _reactor: An L{IReactorThreads} provider which will be passed on to
        L{_WSGIResponse} to schedule calls in the I/O thread.

    @ivar _threadpool: A L{ThreadPool} which will be passed on to
        L{_WSGIResponse} to run the WSGI application object.

    @ivar _application: The WSGI application object.
    """
    isLeaf = True

    def __init__(self, reactor, threadpool, application):
        self._reactor = reactor
        self._threadpool = threadpool
        self._application = application

    def render(self, request):
        """
        Turn the request into the appropriate C{environ} C{dict} suitable to be
        passed to the WSGI application object and then pass it on.

        The WSGI application object is given almost complete control of the
        rendering process.  C{NOT_DONE_YET} will always be returned in order
        and response completion will be dictated by the application object, as
        will the status, headers, and the response body.
        """
        response = _WSGIResponse(self._reactor, self._threadpool, self._application, request)
        response.start()
        return NOT_DONE_YET

    def getChildWithDefault(self, name, request):
        """
        Reject attempts to retrieve a child resource.  All path segments beyond
        the one which refers to this resource are handled by the WSGI
        application object.
        """
        raise RuntimeError('Cannot get IResource children from WSGIResource')

    def putChild(self, path, child):
        """
        Reject attempts to add a child resource to this resource.  The WSGI
        application object handles all path segments beneath this resource, so
        L{IResource} children can never be found.
        """
        raise RuntimeError('Cannot put IResource children under WSGIResource')