import re
import threading
class ConfigMiddleware:
    """
    A WSGI middleware that adds a ``paste.config`` key to the request
    environment, as well as registering the configuration temporarily
    (for the length of the request) with ``paste.CONFIG``.
    """

    def __init__(self, application, config):
        """
        This delegates all requests to `application`, adding a *copy*
        of the configuration `config`.
        """
        self.application = application
        self.config = config

    def __call__(self, environ, start_response):
        global wsgilib
        if wsgilib is None:
            from paste import wsgilib
        popped_config = None
        if 'paste.config' in environ:
            popped_config = environ['paste.config']
        conf = environ['paste.config'] = self.config.copy()
        app_iter = None
        CONFIG.push_thread_config(conf)
        try:
            app_iter = self.application(environ, start_response)
        finally:
            if app_iter is None:
                CONFIG.pop_thread_config(conf)
                if popped_config is not None:
                    environ['paste.config'] = popped_config
        if type(app_iter) in (list, tuple):
            CONFIG.pop_thread_config(conf)
            if popped_config is not None:
                environ['paste.config'] = popped_config
            return app_iter
        else:

            def close_config():
                CONFIG.pop_thread_config(conf)
            new_app_iter = wsgilib.add_close(app_iter, close_config)
            return new_app_iter