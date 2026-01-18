from django.core.exceptions import ImproperlyConfigured
from django.core.paginator import InvalidPage, Paginator
from django.db.models import QuerySet
from django.http import Http404
from django.utils.translation import gettext as _
from django.views.generic.base import ContextMixin, TemplateResponseMixin, View
class MultipleObjectTemplateResponseMixin(TemplateResponseMixin):
    """Mixin for responding with a template and list of objects."""
    template_name_suffix = '_list'

    def get_template_names(self):
        """
        Return a list of template names to be used for the request. Must return
        a list. May not be called if render_to_response is overridden.
        """
        try:
            names = super().get_template_names()
        except ImproperlyConfigured:
            names = []
        if hasattr(self.object_list, 'model'):
            opts = self.object_list.model._meta
            names.append('%s/%s%s.html' % (opts.app_label, opts.model_name, self.template_name_suffix))
        elif not names:
            raise ImproperlyConfigured("%(cls)s requires either a 'template_name' attribute or a get_queryset() method that returns a QuerySet." % {'cls': self.__class__.__name__})
        return names