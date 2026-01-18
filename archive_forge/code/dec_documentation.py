from webob.compat import (
from webob.request import Request
from webob.exc import HTTPException
Creates middleware

        Use this like::

            @wsgify.middleware
            def restrict_ip(req, app, ips):
                if req.remote_addr not in ips:
                    raise webob.exc.HTTPForbidden('Bad IP: %s' % req.remote_addr)
                return app

            @wsgify
            def app(req):
                return 'hi'

            wrapped = restrict_ip(app, ips=['127.0.0.1'])

        Or as a decorator::

            @restrict_ip(ips=['127.0.0.1'])
            @wsgify
            def wrapped_app(req):
                return 'hi'

        Or if you want to write output-rewriting middleware::

            @wsgify.middleware
            def all_caps(req, app):
                resp = req.get_response(app)
                resp.body = resp.body.upper()
                return resp

            wrapped = all_caps(app)

        Note that you must call ``req.get_response(app)`` to get a WebOb
        response object.  If you are not modifying the output, you can just
        return the app.

        As you can see, this method doesn't actually create an application, but
        creates "middleware" that can be bound to an application, along with
        "configuration" (that is, any other keyword arguments you pass when
        binding the application).

        