import configparser
import os
from paste.deploy import loadapp
from gunicorn.app.wsgiapp import WSGIApplication
from gunicorn.config import get_default_config_file
class PasterServerApplication(WSGIApplication):

    def load_config(self):
        self.cfg.set('default_proc_name', config_file)
        if has_logging_config(config_file):
            self.cfg.set('logconfig', config_file)
        if gunicorn_config_file:
            self.load_config_from_file(gunicorn_config_file)
        else:
            default_gunicorn_config_file = get_default_config_file()
            if default_gunicorn_config_file is not None:
                self.load_config_from_file(default_gunicorn_config_file)
        for k, v in local_conf.items():
            if v is not None:
                self.cfg.set(k.lower(), v)

    def load(self):
        return app