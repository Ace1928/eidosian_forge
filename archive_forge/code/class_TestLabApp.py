import urllib.parse
import pytest
from jupyter_server.utils import url_path_join
from jupyterlab_server import LabConfig
from tornado.escape import url_escape
from traitlets import Unicode
from jupyterlab.labapp import LabApp
class TestLabApp(LabApp):
    base_url = '/lab'
    extension_url = '/lab'
    default_url = Unicode('/', help='The default URL to redirect to from `/`')
    lab_config = LabConfig(app_name='JupyterLab Test App', static_dir=str(jp_root_dir), templates_dir=str(jp_template_dir), app_url='/lab', app_settings_dir=str(app_settings_dir), user_settings_dir=str(user_settings_dir), schemas_dir=str(schemas_dir), workspaces_dir=str(workspaces_dir))