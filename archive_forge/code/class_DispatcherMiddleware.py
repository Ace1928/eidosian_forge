from __future__ import annotations
import typing as t
class DispatcherMiddleware:
    """Combine multiple applications as a single WSGI application.
    Requests are dispatched to an application based on the path it is
    mounted under.

    :param app: The WSGI application to dispatch to if the request
        doesn't match a mounted path.
    :param mounts: Maps path prefixes to applications for dispatching.
    """

    def __init__(self, app: WSGIApplication, mounts: dict[str, WSGIApplication] | None=None) -> None:
        self.app = app
        self.mounts = mounts or {}

    def __call__(self, environ: WSGIEnvironment, start_response: StartResponse) -> t.Iterable[bytes]:
        script = environ.get('PATH_INFO', '')
        path_info = ''
        while '/' in script:
            if script in self.mounts:
                app = self.mounts[script]
                break
            script, last_item = script.rsplit('/', 1)
            path_info = f'/{last_item}{path_info}'
        else:
            app = self.mounts.get(script, self.app)
        original_script_name = environ.get('SCRIPT_NAME', '')
        environ['SCRIPT_NAME'] = original_script_name + script
        environ['PATH_INFO'] = path_info
        return app(environ, start_response)