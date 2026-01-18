import six
from genshi.input import ET, HTML, XML
from genshi.output import DocType
from genshi.template.base import Template
from genshi.template.loader import TemplateLoader
from genshi.template.markup import MarkupTemplate
from genshi.template.text import TextTemplate, NewTextTemplate
def _get_render_options(self, format=None, fragment=False):
    kwargs = super(MarkupTemplateEnginePlugin, self)._get_render_options(format, fragment)
    if self.default_doctype and (not fragment):
        kwargs['doctype'] = self.default_doctype
    return kwargs