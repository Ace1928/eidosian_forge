import json
import os
import re
from pathlib import Path
from django.apps import apps
from django.conf import settings
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.template import Context, Engine
from django.urls import translate_url
from django.utils.formats import get_format
from django.utils.http import url_has_allowed_host_and_scheme
from django.utils.translation import check_for_language, get_language
from django.utils.translation.trans_real import DjangoTranslation
from django.views.generic import View
class JavaScriptCatalog(View):
    """
    Return the selected language catalog as a JavaScript library.

    Receive the list of packages to check for translations in the `packages`
    kwarg either from the extra dictionary passed to the path() function or as
    a plus-sign delimited string from the request. Default is 'django.conf'.

    You can override the gettext domain for this view, but usually you don't
    want to do that as JavaScript messages go to the djangojs domain. This
    might be needed if you deliver your JavaScript source from Django templates.
    """
    domain = 'djangojs'
    packages = None

    def get(self, request, *args, **kwargs):
        locale = get_language()
        domain = kwargs.get('domain', self.domain)
        packages = kwargs.get('packages', '')
        packages = packages.split('+') if packages else self.packages
        paths = self.get_paths(packages) if packages else None
        self.translation = DjangoTranslation(locale, domain=domain, localedirs=paths)
        context = self.get_context_data(**kwargs)
        return self.render_to_response(context)

    def get_paths(self, packages):
        allowable_packages = {app_config.name: app_config for app_config in apps.get_app_configs()}
        app_configs = [allowable_packages[p] for p in packages if p in allowable_packages]
        if len(app_configs) < len(packages):
            excluded = [p for p in packages if p not in allowable_packages]
            raise ValueError('Invalid package(s) provided to JavaScriptCatalog: %s' % ','.join(excluded))
        return [os.path.join(app.path, 'locale') for app in app_configs]

    @property
    def _num_plurals(self):
        """
        Return the number of plurals for this catalog language, or 2 if no
        plural string is available.
        """
        match = re.search('nplurals=\\s*(\\d+)', self._plural_string or '')
        if match:
            return int(match[1])
        return 2

    @property
    def _plural_string(self):
        """
        Return the plural string (including nplurals) for this catalog language,
        or None if no plural string is available.
        """
        if '' in self.translation._catalog:
            for line in self.translation._catalog[''].split('\n'):
                if line.startswith('Plural-Forms:'):
                    return line.split(':', 1)[1].strip()
        return None

    def get_plural(self):
        plural = self._plural_string
        if plural is not None:
            plural = [el.strip() for el in plural.split(';') if el.strip().startswith('plural=')][0].split('=', 1)[1]
        return plural

    def get_catalog(self):
        pdict = {}
        catalog = {}
        translation = self.translation
        seen_keys = set()
        while True:
            for key, value in translation._catalog.items():
                if key == '' or key in seen_keys:
                    continue
                if isinstance(key, str):
                    catalog[key] = value
                elif isinstance(key, tuple):
                    msgid, cnt = key
                    pdict.setdefault(msgid, {})[cnt] = value
                else:
                    raise TypeError(key)
                seen_keys.add(key)
            if translation._fallback:
                translation = translation._fallback
            else:
                break
        num_plurals = self._num_plurals
        for k, v in pdict.items():
            catalog[k] = [v.get(i, '') for i in range(num_plurals)]
        return catalog

    def get_context_data(self, **kwargs):
        return {'catalog': self.get_catalog(), 'formats': get_formats(), 'plural': self.get_plural()}

    def render_to_response(self, context, **response_kwargs):

        def indent(s):
            return s.replace('\n', '\n  ')
        with builtin_template_path('i18n_catalog.js').open(encoding='utf-8') as fh:
            template = Engine().from_string(fh.read())
        context['catalog_str'] = indent(json.dumps(context['catalog'], sort_keys=True, indent=2)) if context['catalog'] else None
        context['formats_str'] = indent(json.dumps(context['formats'], sort_keys=True, indent=2))
        return HttpResponse(template.render(Context(context)), 'text/javascript; charset="utf-8"')