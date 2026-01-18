from traitlets import default
from traitlets.config import Config
from .templateexporter import TemplateExporter
@default('template_name')
def _template_name_default(self):
    return 'asciidoc'