import os
from jupyter_server.serverapp import ServerApp
from traitlets import default
from .labapp import LabApp
class SingleUserLabApp(SingleUserServerApp):

    @default('default_url')
    def _default_url(self):
        return '/lab'

    def find_server_extensions(self):
        """unconditionally enable jupyterlab server extension

        never called if using legacy SingleUserNotebookApp
        """
        super().find_server_extensions()
        self.jpserver_extensions[LabApp.get_extension_package()] = True