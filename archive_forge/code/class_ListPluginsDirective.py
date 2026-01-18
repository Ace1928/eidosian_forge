import inspect
from docutils import nodes
from docutils.parsers import rst
from docutils.parsers.rst import directives
from docutils.statemachine import ViewList
from sphinx.util import logging
from sphinx.util.nodes import nested_parse_with_titles
from stevedore import extension
class ListPluginsDirective(rst.Directive):
    """Present a simple list of the plugins in a namespace."""
    option_spec = {'class': directives.class_option, 'detailed': directives.flag, 'titlecase': directives.flag, 'overline-style': directives.single_char_or_unicode, 'underline-style': directives.single_char_or_unicode}
    has_content = True

    def run(self):
        namespace = ' '.join(self.content).strip()
        LOG.info('documenting plugins from %r' % namespace)
        overline_style = self.options.get('overline-style', '')
        underline_style = self.options.get('underline-style', '=')

        def report_load_failure(mgr, ep, err):
            LOG.warning(u'Failed to load %s: %s' % (ep.module, err))
        mgr = extension.ExtensionManager(namespace, on_load_failure_callback=report_load_failure)
        result = ViewList()
        titlecase = 'titlecase' in self.options
        if 'detailed' in self.options:
            data = _detailed_list(mgr, over=overline_style, under=underline_style, titlecase=titlecase)
        else:
            data = _simple_list(mgr)
        for text, source in data:
            for line in text.splitlines():
                result.append(line, source)
        node = nodes.section()
        node.document = self.state.document
        nested_parse_with_titles(self.state, result, node)
        return node.children