import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from jupyter_server.base.handlers import APIHandler
from jupyter_server.extension.handler import ExtensionHandlerMixin
from tornado import gen, web
from tornado.concurrent import run_on_executor
from jupyterlab.commands import AppOptions, _ensure_options, build, build_check, clean
class BuildHandler(ExtensionHandlerMixin, APIHandler):

    def initialize(self, builder=None, name=None):
        super().initialize(name=name)
        self.builder = builder

    @web.authenticated
    @gen.coroutine
    def get(self):
        data = (yield self.builder.get_status())
        self.finish(json.dumps(data))

    @web.authenticated
    @gen.coroutine
    def delete(self):
        self.log.warning('Canceling build')
        try:
            yield self.builder.cancel()
        except Exception as e:
            raise web.HTTPError(500, str(e)) from None
        self.set_status(204)

    @web.authenticated
    @gen.coroutine
    def post(self):
        self.log.debug('Starting build')
        try:
            yield self.builder.build()
        except Exception as e:
            raise web.HTTPError(500, str(e)) from None
        if self.builder.canceled:
            raise web.HTTPError(400, 'Build canceled')
        self.log.debug('Build succeeded')
        self.set_status(200)