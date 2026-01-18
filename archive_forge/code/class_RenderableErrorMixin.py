import json
from collections import UserList
from django.conf import settings
from django.core.exceptions import ValidationError
from django.forms.renderers import get_default_renderer
from django.utils import timezone
from django.utils.html import escape, format_html_join
from django.utils.safestring import mark_safe
from django.utils.translation import gettext_lazy as _
class RenderableErrorMixin(RenderableMixin):

    def as_json(self, escape_html=False):
        return json.dumps(self.get_json_data(escape_html))

    def as_text(self):
        return self.render(self.template_name_text)

    def as_ul(self):
        return self.render(self.template_name_ul)