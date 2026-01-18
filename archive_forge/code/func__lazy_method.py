from functools import partial
from django.db.models.utils import make_model_tuple
from django.dispatch import Signal
def _lazy_method(self, method, apps, receiver, sender, **kwargs):
    from django.db.models.options import Options
    partial_method = partial(method, receiver, **kwargs)
    if isinstance(sender, str):
        apps = apps or Options.default_apps
        apps.lazy_model_operation(partial_method, make_model_tuple(sender))
    else:
        return partial_method(sender)