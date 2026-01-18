import os
from traitlets.config.application import Application
from IPython.core.application import (
from IPython.core.profiledir import ProfileDir
from IPython.utils.importstring import import_item
from IPython.paths import get_ipython_dir, get_ipython_package_dir
from traitlets import Unicode, Bool, Dict, observe
def _import_app(self, app_path):
    """import an app class"""
    app = None
    name = app_path.rsplit('.', 1)[-1]
    try:
        app = import_item(app_path)
    except ImportError:
        self.log.info("Couldn't import %s, config file will be excluded", name)
    except Exception:
        self.log.warning('Unexpected error importing %s', name, exc_info=True)
    return app