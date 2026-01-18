import mimetypes
from typing import Optional
import traitlets
from traitlets.config import Config
from nbconvert.filters.markdown_mistune import IPythonRenderer, MarkdownWithMath
from nbconvert.exporters.html import HTMLExporter
from nbconvert.exporters.templateexporter import TemplateExporter
from nbconvert.filters.highlight import Highlight2HTML
from .static_file_handler import TemplateStaticFileHandler
from .utils import create_include_assets_functions
@traitlets.validate('template_name')
def _validate_template_name(self, template_name):
    if template_name.value == 'classic':
        self.log.warning('"classic" template support will be removed in Voila 1.0.0, please use the "lab" template instead with the "--show-margins" option for a similar look')
    return template_name.value