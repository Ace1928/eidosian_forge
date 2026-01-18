from django.template import Library, Node, TemplateSyntaxError
from django.utils import formats
class LocalizeNode(Node):

    def __init__(self, nodelist, use_l10n):
        self.nodelist = nodelist
        self.use_l10n = use_l10n

    def __repr__(self):
        return '<%s>' % self.__class__.__name__

    def render(self, context):
        old_setting = context.use_l10n
        context.use_l10n = self.use_l10n
        output = self.nodelist.render(context)
        context.use_l10n = old_setting
        return output