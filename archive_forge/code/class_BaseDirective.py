import re
class BaseDirective:
    parser = DirectiveParser
    directive_pattern = None

    def __init__(self, plugins):
        self._methods = {}
        self.__plugins = plugins

    def register(self, name, fn):
        self._methods[name] = fn

    def parse_method(self, block, m, state):
        _type = self.parser.parse_type(m)
        method = self._methods.get(_type)
        if method:
            try:
                token = method(block, m, state)
            except ValueError as e:
                token = {'type': 'block_error', 'raw': str(e)}
        else:
            text = m.group(0)
            token = {'type': 'block_error', 'raw': text}
        if isinstance(token, list):
            for tok in token:
                state.append_token(tok)
        else:
            state.append_token(token)
        return token

    def parse_directive(self, block, m, state):
        raise NotImplementedError()

    def register_block_parser(self, md, before=None):
        md.block.register(self.parser.name, self.directive_pattern, self.parse_directive, before=before)

    def __call__(self, md):
        for plugin in self.__plugins:
            plugin.parser = self.parser
            plugin(self, md)