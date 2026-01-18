from docutils import nodes, utils
from docutils.parsers.rst import directives
from docutils.parsers.rst.languages import en as _fallback_language_module
from docutils.utils.code_analyzer import Lexer, LexerError
class CustomRole:
    """
    Wrapper for custom interpreted text roles.
    """

    def __init__(self, role_name, base_role, options={}, content=[]):
        self.name = role_name
        self.base_role = base_role
        self.options = None
        if hasattr(base_role, 'options'):
            self.options = base_role.options
        self.content = None
        if hasattr(base_role, 'content'):
            self.content = base_role.content
        self.supplied_options = options
        self.supplied_content = content

    def __call__(self, role, rawtext, text, lineno, inliner, options={}, content=[]):
        opts = self.supplied_options.copy()
        opts.update(options)
        cont = list(self.supplied_content)
        if cont and content:
            cont += '\n'
        cont.extend(content)
        return self.base_role(role, rawtext, text, lineno, inliner, options=opts, content=cont)