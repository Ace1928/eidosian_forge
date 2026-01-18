from suds import *
from logging import getLogger
class PluginDomain:
    """
    The plugin domain.

    @ivar ctx: A context.
    @type ctx: L{Context}
    @ivar plugins: A list of plugins (targets).
    @type plugins: list

    """

    def __init__(self, ctx, plugins):
        self.ctx = ctx
        self.plugins = plugins

    def __getattr__(self, name):
        return Method(name, self)