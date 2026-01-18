import string
import sys
import types
import cherrypy
class LateParamPageHandler(PageHandler):
    """When passing cherrypy.request.params to the page handler, we do not
    want to capture that dict too early; we want to give tools like the
    decoding tool a chance to modify the params dict in-between the lookup
    of the handler and the actual calling of the handler. This subclass
    takes that into account, and allows request.params to be 'bound late'
    (it's more complicated than that, but that's the effect).
    """

    @property
    def kwargs(self):
        """Page handler kwargs (with cherrypy.request.params copied in)."""
        kwargs = cherrypy.serving.request.params.copy()
        if self._kwargs:
            kwargs.update(self._kwargs)
        return kwargs

    @kwargs.setter
    def kwargs(self, kwargs):
        cherrypy.serving.request.kwargs = kwargs
        self._kwargs = kwargs