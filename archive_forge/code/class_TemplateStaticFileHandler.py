import os
import re
import tornado.web
from .paths import collect_static_paths
class TemplateStaticFileHandler(tornado.web.StaticFileHandler):
    """Static file handler that serves the static files for the template system.

    URL paths should be of the form <`template_name>/static/<path>`

    A url such as lab/static/voila.js can be translated to a real path such
    /my/prefix/jupyter/voila/templates/base/static/voila.js
    Meaning the url portion is not part of the real (absolute path)

    For this system, we don't need to use the root, since this is handled in the
    paths module.
    """

    def initialize(self):
        super().initialize(path='/fake-root/voila-template-system/')

    def parse_url_path(self, path):
        template, static, _ignore = path.split('/', 2)
        assert static == 'static'
        self.roots = collect_static_paths(['voila', 'nbconvert'], template)
        return super().parse_url_path(path)

    def validate_absolute_path(self, root: str, absolute_path: str):
        last_exception = None
        for root in self.roots:
            try:
                return super().validate_absolute_path(root, absolute_path)
            except tornado.web.HTTPError as e:
                last_exception = e
        assert last_exception
        raise last_exception

    @classmethod
    def get_absolute_path(cls, root, path):
        template, static, relpath = os.path.normpath(path).split(os.path.sep, 2)
        assert static == 'static'
        roots = collect_static_paths(['voila', 'nbconvert'], template)
        for root in roots:
            abspath = os.path.abspath(os.path.join(root, relpath))
            if os.path.exists(abspath):
                return abspath
                break
        return abspath