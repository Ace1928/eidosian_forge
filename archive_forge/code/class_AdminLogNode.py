from django import template
class AdminLogNode(template.Node):

    def __init__(self, limit, varname, user):
        self.limit = limit
        self.varname = varname
        self.user = user

    def __repr__(self):
        return '<GetAdminLog Node>'

    def render(self, context):
        entries = context['log_entries']
        if self.user is not None:
            user_id = self.user
            if not user_id.isdigit():
                user_id = context[self.user].pk
            entries = entries.filter(user__pk=user_id)
        context[self.varname] = entries[:int(self.limit)]
        return ''