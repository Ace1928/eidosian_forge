import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from jupyter_server.base.handlers import APIHandler
from jupyter_server.extension.handler import ExtensionHandlerMixin
from tornado import gen, web
from tornado.concurrent import run_on_executor
from jupyterlab.commands import AppOptions, _ensure_options, build, build_check, clean
@run_on_executor
def _run_build_check(self, app_dir, logger, core_config, labextensions_path):
    return build_check(app_options=AppOptions(app_dir=app_dir, logger=logger, core_config=core_config, labextensions_path=labextensions_path))