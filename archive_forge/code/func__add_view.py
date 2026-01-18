from django.conf import settings
from django.contrib import admin, messages
from django.contrib.admin.options import IS_POPUP_VAR
from django.contrib.admin.utils import unquote
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.forms import (
from django.contrib.auth.models import Group, User
from django.core.exceptions import PermissionDenied
from django.db import router, transaction
from django.http import Http404, HttpResponseRedirect
from django.template.response import TemplateResponse
from django.urls import path, reverse
from django.utils.decorators import method_decorator
from django.utils.html import escape
from django.utils.translation import gettext
from django.utils.translation import gettext_lazy as _
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.debug import sensitive_post_parameters
def _add_view(self, request, form_url='', extra_context=None):
    if not self.has_change_permission(request):
        if self.has_add_permission(request) and settings.DEBUG:
            raise Http404('Your user does not have the "Change user" permission. In order to add users, Django requires that your user account have both the "Add user" and "Change user" permissions set.')
        raise PermissionDenied
    if extra_context is None:
        extra_context = {}
    username_field = self.opts.get_field(self.model.USERNAME_FIELD)
    defaults = {'auto_populated_fields': (), 'username_help_text': username_field.help_text}
    extra_context.update(defaults)
    return super().add_view(request, form_url, extra_context)