import oslo_reports.views.jinja_view as jv
class ProcessView(jv.JinjaView):
    """A Process View

    This view displays process models defined by
    :class:`oslo_reports.models.process.ProcessModel`
    """
    VIEW_TEXT = "Process {{ pid }} (under {{ parent_pid }}) [ run by: {{ username }} ({{ uids.real|default('unknown uid') }}), state: {{ state }} ]\n{% for child in children %}    {{ child }}{% endfor %}"