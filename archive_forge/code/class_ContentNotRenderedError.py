from django.http import HttpResponse
from .loader import get_template, select_template
class ContentNotRenderedError(Exception):
    pass