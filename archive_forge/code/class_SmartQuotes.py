import re
import sys
import time
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
from docutils.utils import smartquotes
class SmartQuotes(Transform):
    """
    Replace ASCII quotation marks with typographic form.

    Also replace multiple dashes with em-dash/en-dash characters.
    """
    default_priority = 850
    nodes_to_skip = (nodes.FixedTextElement, nodes.Special)
    'Do not apply "smartquotes" to instances of these block-level nodes.'
    literal_nodes = (nodes.image, nodes.literal, nodes.math, nodes.raw, nodes.problematic)
    'Do not change quotes in instances of these inline nodes.'
    smartquotes_action = 'qDe'
    'Setting to select smartquote transformations.\n\n    The default \'qDe\' educates normal quote characters: (", \'),\n    em- and en-dashes (---, --) and ellipses (...).\n    '

    def __init__(self, document, startnode):
        Transform.__init__(self, document, startnode=startnode)
        self.unsupported_languages = set()

    def get_tokens(self, txtnodes):
        texttype = {True: 'literal', False: 'plain'}
        for txtnode in txtnodes:
            nodetype = texttype[isinstance(txtnode.parent, self.literal_nodes)]
            yield (nodetype, txtnode.astext())

    def apply(self):
        smart_quotes = self.document.settings.smart_quotes
        if not smart_quotes:
            return
        try:
            alternative = smart_quotes.startswith('alt')
        except AttributeError:
            alternative = False
        document_language = self.document.settings.language_code
        lc_smartquotes = self.document.settings.smartquotes_locales
        if lc_smartquotes:
            smartquotes.smartchars.quotes.update(dict(lc_smartquotes))
        for node in self.document.traverse(nodes.TextElement):
            if isinstance(node, self.nodes_to_skip):
                continue
            if isinstance(node.parent, nodes.TextElement):
                continue
            txtnodes = [txtnode for txtnode in node.traverse(nodes.Text) if not isinstance(txtnode.parent, nodes.option_string)]
            lang = node.get_language_code(document_language)
            if alternative:
                if '-x-altquot' in lang:
                    lang = lang.replace('-x-altquot', '')
                else:
                    lang += '-x-altquot'
            for tag in utils.normalize_language_tag(lang):
                if tag in smartquotes.smartchars.quotes:
                    lang = tag
                    break
            else:
                if lang not in self.unsupported_languages:
                    self.document.reporter.warning('No smart quotes defined for language "%s".' % lang, base_node=node)
                self.unsupported_languages.add(lang)
                lang = ''
            teacher = smartquotes.educate_tokens(self.get_tokens(txtnodes), attr=self.smartquotes_action, language=lang)
            for txtnode, newtext in zip(txtnodes, teacher):
                txtnode.parent.replace(txtnode, nodes.Text(newtext, rawsource=txtnode.rawsource))
        self.unsupported_languages = set()