from .compat import escape
from .jsonify import encode
class JinjaRenderer(object):
    """
        Defines the builtin ``Jinja`` renderer.
        """

    def __init__(self, path, extra_vars):
        self.env = Environment(loader=FileSystemLoader(path))
        self.extra_vars = extra_vars

    def render(self, template_path, namespace):
        """
            Implements ``Jinja`` rendering.
            """
        template = self.env.get_template(template_path)
        return template.render(self.extra_vars.make_ns(namespace))