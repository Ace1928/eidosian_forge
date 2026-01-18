from collections import defaultdict
from django.apps import apps
from django.db import models
from django.db.models import Q
from django.utils.translation import gettext_lazy as _
class ContentTypeManager(models.Manager):
    use_in_migrations = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = {}

    def get_by_natural_key(self, app_label, model):
        try:
            ct = self._cache[self.db][app_label, model]
        except KeyError:
            ct = self.get(app_label=app_label, model=model)
            self._add_to_cache(self.db, ct)
        return ct

    def _get_opts(self, model, for_concrete_model):
        if for_concrete_model:
            model = model._meta.concrete_model
        return model._meta

    def _get_from_cache(self, opts):
        key = (opts.app_label, opts.model_name)
        return self._cache[self.db][key]

    def get_for_model(self, model, for_concrete_model=True):
        """
        Return the ContentType object for a given model, creating the
        ContentType if necessary. Lookups are cached so that subsequent lookups
        for the same model don't hit the database.
        """
        opts = self._get_opts(model, for_concrete_model)
        try:
            return self._get_from_cache(opts)
        except KeyError:
            pass
        try:
            ct = self.get(app_label=opts.app_label, model=opts.model_name)
        except self.model.DoesNotExist:
            ct, created = self.get_or_create(app_label=opts.app_label, model=opts.model_name)
        self._add_to_cache(self.db, ct)
        return ct

    def get_for_models(self, *models, for_concrete_models=True):
        """
        Given *models, return a dictionary mapping {model: content_type}.
        """
        results = {}
        needed_models = defaultdict(set)
        needed_opts = defaultdict(list)
        for model in models:
            opts = self._get_opts(model, for_concrete_models)
            try:
                ct = self._get_from_cache(opts)
            except KeyError:
                needed_models[opts.app_label].add(opts.model_name)
                needed_opts[opts].append(model)
            else:
                results[model] = ct
        if needed_opts:
            condition = Q(*(Q(('app_label', app_label), ('model__in', models)) for app_label, models in needed_models.items()), _connector=Q.OR)
            cts = self.filter(condition)
            for ct in cts:
                opts_models = needed_opts.pop(ct._meta.apps.get_model(ct.app_label, ct.model)._meta, [])
                for model in opts_models:
                    results[model] = ct
                self._add_to_cache(self.db, ct)
        for opts, opts_models in needed_opts.items():
            ct = self.create(app_label=opts.app_label, model=opts.model_name)
            self._add_to_cache(self.db, ct)
            for model in opts_models:
                results[model] = ct
        return results

    def get_for_id(self, id):
        """
        Lookup a ContentType by ID. Use the same shared cache as get_for_model
        (though ContentTypes are not created on-the-fly by get_by_id).
        """
        try:
            ct = self._cache[self.db][id]
        except KeyError:
            ct = self.get(pk=id)
            self._add_to_cache(self.db, ct)
        return ct

    def clear_cache(self):
        """
        Clear out the content-type cache.
        """
        self._cache.clear()

    def _add_to_cache(self, using, ct):
        """Insert a ContentType into the cache."""
        key = (ct.app_label, ct.model)
        self._cache.setdefault(using, {})[key] = ct
        self._cache.setdefault(using, {})[ct.id] = ct