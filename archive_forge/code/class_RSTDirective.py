import re
from ._base import DirectiveParser, BaseDirective
class RSTDirective(BaseDirective):
    """A RST style of directive syntax is inspired by reStructuredText.
    The syntax is very powerful that you can define a lot of custom
    features on your own. The syntax looks like:

    .. code-block:: text

        .. directive-type:: directive value
           :option-key: option value
           :option-key: option value

           content text here

    To use ``RSTDirective``, developers can add it into plugin list in
    the :class:`Markdown` instance:

    .. code-block:: python

        import mistune
        from mistune.directives import RSTDirective, Admonition

        md = mistune.create_markdown(plugins=[
            # ...
            RSTDirective([Admonition()]),
        ])
    """
    parser = RSTParser
    directive_pattern = '^\\.\\. +[a-zA-Z0-9_-]+\\:\\:'

    def parse_directive(self, block, m, state):
        m = _directive_re.match(state.src, state.cursor)
        if not m:
            return
        self.parse_method(block, m, state)
        return m.end()

    def __call__(self, md):
        super(RSTDirective, self).__call__(md)
        self.register_block_parser(md)