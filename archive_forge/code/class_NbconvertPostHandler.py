import io
import os
import sys
import zipfile
from anyio.to_thread import run_sync
from jupyter_core.utils import ensure_async
from nbformat import from_dict
from tornado import web
from tornado.log import app_log
from jupyter_server.auth.decorator import authorized
from ..base.handlers import FilesRedirectHandler, JupyterHandler, path_regex
class NbconvertPostHandler(JupyterHandler):
    """An nbconvert post handler."""
    SUPPORTED_METHODS = ('POST',)
    auth_resource = AUTH_RESOURCE

    @web.authenticated
    @authorized
    async def post(self, format):
        """Convert a notebook file to a desired format."""
        exporter = get_exporter(format, config=self.config)
        model = self.get_json_body()
        assert model is not None
        name = model.get('name', 'notebook.ipynb')
        nbnode = from_dict(model['content'])
        try:
            output, resources = await run_sync(lambda: exporter.from_notebook_node(nbnode, resources={'metadata': {'name': name[:name.rfind('.')]}, 'config_dir': self.application.settings['config_dir']}))
        except Exception as e:
            raise web.HTTPError(500, 'nbconvert failed: %s' % e) from e
        if respond_zip(self, name, output, resources):
            return
        if exporter.output_mimetype:
            self.set_header('Content-Type', '%s; charset=utf-8' % exporter.output_mimetype)
        self.finish(output)