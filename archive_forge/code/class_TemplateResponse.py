from django.http import HttpResponse
from .loader import get_template, select_template
class TemplateResponse(SimpleTemplateResponse):
    rendering_attrs = SimpleTemplateResponse.rendering_attrs + ['_request']

    def __init__(self, request, template, context=None, content_type=None, status=None, charset=None, using=None, headers=None):
        super().__init__(template, context, content_type, status, charset, using, headers=headers)
        self._request = request