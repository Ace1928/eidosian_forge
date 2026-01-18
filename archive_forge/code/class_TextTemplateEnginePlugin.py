import six
from genshi.input import ET, HTML, XML
from genshi.output import DocType
from genshi.template.base import Template
from genshi.template.loader import TemplateLoader
from genshi.template.markup import MarkupTemplate
from genshi.template.text import TextTemplate, NewTextTemplate
class TextTemplateEnginePlugin(AbstractTemplateEnginePlugin):
    """Implementation of the plugin API for text templates."""
    template_class = TextTemplate
    extension = '.txt'
    default_format = 'text'

    def __init__(self, extra_vars_func=None, options=None):
        if options is None:
            options = {}
        new_syntax = options.get('genshi.new_text_syntax')
        if isinstance(new_syntax, six.string_types):
            new_syntax = new_syntax.lower() in ('1', 'on', 'yes', 'true')
        if new_syntax:
            self.template_class = NewTextTemplate
        AbstractTemplateEnginePlugin.__init__(self, extra_vars_func, options)