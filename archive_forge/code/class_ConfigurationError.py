import six
from genshi.input import ET, HTML, XML
from genshi.output import DocType
from genshi.template.base import Template
from genshi.template.loader import TemplateLoader
from genshi.template.markup import MarkupTemplate
from genshi.template.text import TextTemplate, NewTextTemplate
class ConfigurationError(ValueError):
    """Exception raised when invalid plugin options are encountered."""