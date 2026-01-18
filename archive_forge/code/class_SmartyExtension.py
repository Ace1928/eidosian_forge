from __future__ import annotations
from . import Extension
from ..inlinepatterns import HtmlInlineProcessor, HTML_RE
from ..treeprocessors import InlineProcessor
from ..util import Registry
from typing import TYPE_CHECKING, Sequence
class SmartyExtension(Extension):
    """ Add Smarty to Markdown. """

    def __init__(self, **kwargs):
        self.config = {'smart_quotes': [True, 'Educate quotes'], 'smart_angled_quotes': [False, 'Educate angled quotes'], 'smart_dashes': [True, 'Educate dashes'], 'smart_ellipses': [True, 'Educate ellipses'], 'substitutions': [{}, 'Overwrite default substitutions']}
        ' Default configuration options. '
        super().__init__(**kwargs)
        self.substitutions: dict[str, str] = dict(substitutions)
        self.substitutions.update(self.getConfig('substitutions', default={}))

    def _addPatterns(self, md: Markdown, patterns: Sequence[tuple[str, Sequence[int | str | etree.Element]]], serie: str, priority: int):
        for ind, pattern in enumerate(patterns):
            pattern += (md,)
            pattern = SubstituteTextPattern(*pattern)
            name = 'smarty-%s-%d' % (serie, ind)
            self.inlinePatterns.register(pattern, name, priority - ind)

    def educateDashes(self, md: Markdown) -> None:
        emDashesPattern = SubstituteTextPattern('(?<!-)---(?!-)', (self.substitutions['mdash'],), md)
        enDashesPattern = SubstituteTextPattern('(?<!-)--(?!-)', (self.substitutions['ndash'],), md)
        self.inlinePatterns.register(emDashesPattern, 'smarty-em-dashes', 50)
        self.inlinePatterns.register(enDashesPattern, 'smarty-en-dashes', 45)

    def educateEllipses(self, md: Markdown) -> None:
        ellipsesPattern = SubstituteTextPattern('(?<!\\.)\\.{3}(?!\\.)', (self.substitutions['ellipsis'],), md)
        self.inlinePatterns.register(ellipsesPattern, 'smarty-ellipses', 10)

    def educateAngledQuotes(self, md: Markdown) -> None:
        leftAngledQuotePattern = SubstituteTextPattern('\\<\\<', (self.substitutions['left-angle-quote'],), md)
        rightAngledQuotePattern = SubstituteTextPattern('\\>\\>', (self.substitutions['right-angle-quote'],), md)
        self.inlinePatterns.register(leftAngledQuotePattern, 'smarty-left-angle-quotes', 40)
        self.inlinePatterns.register(rightAngledQuotePattern, 'smarty-right-angle-quotes', 35)

    def educateQuotes(self, md: Markdown) -> None:
        lsquo = self.substitutions['left-single-quote']
        rsquo = self.substitutions['right-single-quote']
        ldquo = self.substitutions['left-double-quote']
        rdquo = self.substitutions['right-double-quote']
        patterns = ((singleQuoteStartRe, (rsquo,)), (doubleQuoteStartRe, (rdquo,)), (doubleQuoteSetsRe, (ldquo + lsquo,)), (singleQuoteSetsRe, (lsquo + ldquo,)), (decadeAbbrRe, (rsquo,)), (openingSingleQuotesRegex, (1, lsquo)), (closingSingleQuotesRegex, (rsquo,)), (closingSingleQuotesRegex2, (rsquo, 1)), (remainingSingleQuotesRegex, (lsquo,)), (openingDoubleQuotesRegex, (1, ldquo)), (closingDoubleQuotesRegex, (rdquo,)), (closingDoubleQuotesRegex2, (rdquo,)), (remainingDoubleQuotesRegex, (ldquo,)))
        self._addPatterns(md, patterns, 'quotes', 30)

    def extendMarkdown(self, md):
        configs = self.getConfigs()
        self.inlinePatterns: Registry[inlinepatterns.InlineProcessor] = Registry()
        if configs['smart_ellipses']:
            self.educateEllipses(md)
        if configs['smart_quotes']:
            self.educateQuotes(md)
        if configs['smart_angled_quotes']:
            self.educateAngledQuotes(md)
            md.inlinePatterns.register(HtmlInlineProcessor(HTML_STRICT_RE, md), 'html', 90)
        if configs['smart_dashes']:
            self.educateDashes(md)
        inlineProcessor = InlineProcessor(md)
        inlineProcessor.inlinePatterns = self.inlinePatterns
        md.treeprocessors.register(inlineProcessor, 'smarty', 6)
        md.ESCAPED_CHARS.extend(['"', "'"])