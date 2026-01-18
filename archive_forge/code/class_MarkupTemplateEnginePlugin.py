import six
from genshi.input import ET, HTML, XML
from genshi.output import DocType
from genshi.template.base import Template
from genshi.template.loader import TemplateLoader
from genshi.template.markup import MarkupTemplate
from genshi.template.text import TextTemplate, NewTextTemplate
class MarkupTemplateEnginePlugin(AbstractTemplateEnginePlugin):
    """Implementation of the plugin API for markup templates."""
    template_class = MarkupTemplate
    extension = '.html'

    def __init__(self, extra_vars_func=None, options=None):
        AbstractTemplateEnginePlugin.__init__(self, extra_vars_func, options)
        default_doctype = self.options.get('genshi.default_doctype')
        if default_doctype:
            doctype = DocType.get(default_doctype)
            if doctype is None:
                raise ConfigurationError('Unknown doctype %r' % default_doctype)
            self.default_doctype = doctype
        else:
            self.default_doctype = None
        format = self.options.get('genshi.default_format', 'html').lower()
        if format not in ('html', 'xhtml', 'xml', 'text'):
            raise ConfigurationError('Unknown output format %r' % format)
        self.default_format = format

    def _get_render_options(self, format=None, fragment=False):
        kwargs = super(MarkupTemplateEnginePlugin, self)._get_render_options(format, fragment)
        if self.default_doctype and (not fragment):
            kwargs['doctype'] = self.default_doctype
        return kwargs

    def transform(self, info, template):
        """Render the output to an event stream."""
        data = {'ET': ET, 'HTML': HTML, 'XML': XML}
        if self.get_extra_vars:
            data.update(self.get_extra_vars())
        data.update(info)
        return super(MarkupTemplateEnginePlugin, self).transform(data, template)