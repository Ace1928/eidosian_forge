from oslo_reports.views import jinja_view as jv
class StackTraceView(jv.JinjaView):
    """A Stack Trace View

    This view displays stack trace models defined by
    :class:`oslo_reports.models.threading.StackTraceModel`
    """
    VIEW_TEXT = '{% if root_exception is not none %}Exception: {{ root_exception }}\n------------------------------------\n\n{% endif %}{% for line in lines %}\n{{ line.filename }}:{{ line.line }} in {{ line.name }}\n    {% if line.code is not none %}`{{ line.code }}`{% else %}(source not found){% endif %}\n{% else %}\nNo Traceback!\n{% endfor %}'