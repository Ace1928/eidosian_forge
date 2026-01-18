import zoneinfo
from datetime import datetime
from datetime import timezone as datetime_timezone
from datetime import tzinfo
from django.template import Library, Node, TemplateSyntaxError
from django.utils import timezone
class LocalTimeNode(Node):
    """
    Template node class used by ``localtime_tag``.
    """

    def __init__(self, nodelist, use_tz):
        self.nodelist = nodelist
        self.use_tz = use_tz

    def render(self, context):
        old_setting = context.use_tz
        context.use_tz = self.use_tz
        output = self.nodelist.render(context)
        context.use_tz = old_setting
        return output