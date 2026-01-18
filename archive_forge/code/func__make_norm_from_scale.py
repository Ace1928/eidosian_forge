import base64
from collections.abc import Sized, Sequence, Mapping
import functools
import importlib
import inspect
import io
import itertools
from numbers import Real
import re
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import matplotlib as mpl
import numpy as np
from matplotlib import _api, _cm, cbook, scale
from ._color_data import BASE_COLORS, TABLEAU_COLORS, CSS4_COLORS, XKCD_COLORS
@functools.cache
def _make_norm_from_scale(scale_cls, scale_args, scale_kwargs_items, base_norm_cls, bound_init_signature):
    """
    Helper for `make_norm_from_scale`.

    This function is split out to enable caching (in particular so that
    different unpickles reuse the same class).  In order to do so,

    - ``functools.partial`` *scale_cls* is expanded into ``func, args, kwargs``
      to allow memoizing returned norms (partial instances always compare
      unequal, but we can check identity based on ``func, args, kwargs``;
    - *init* is replaced by *init_signature*, as signatures are picklable,
      unlike to arbitrary lambdas.
    """

    class Norm(base_norm_cls):

        def __reduce__(self):
            cls = type(self)
            try:
                if cls is getattr(importlib.import_module(cls.__module__), cls.__qualname__):
                    return (_create_empty_object_of_class, (cls,), vars(self))
            except (ImportError, AttributeError):
                pass
            return (_picklable_norm_constructor, (scale_cls, scale_args, scale_kwargs_items, base_norm_cls, bound_init_signature), vars(self))

        def __init__(self, *args, **kwargs):
            ba = bound_init_signature.bind(*args, **kwargs)
            ba.apply_defaults()
            super().__init__(**{k: ba.arguments.pop(k) for k in ['vmin', 'vmax', 'clip']})
            self._scale = functools.partial(scale_cls, *scale_args, **dict(scale_kwargs_items))(axis=None, **ba.arguments)
            self._trf = self._scale.get_transform()
        __init__.__signature__ = bound_init_signature.replace(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD), *bound_init_signature.parameters.values()])

        def __call__(self, value, clip=None):
            value, is_scalar = self.process_value(value)
            if self.vmin is None or self.vmax is None:
                self.autoscale_None(value)
            if self.vmin > self.vmax:
                raise ValueError('vmin must be less or equal to vmax')
            if self.vmin == self.vmax:
                return np.full_like(value, 0)
            if clip is None:
                clip = self.clip
            if clip:
                value = np.clip(value, self.vmin, self.vmax)
            t_value = self._trf.transform(value).reshape(np.shape(value))
            t_vmin, t_vmax = self._trf.transform([self.vmin, self.vmax])
            if not np.isfinite([t_vmin, t_vmax]).all():
                raise ValueError('Invalid vmin or vmax')
            t_value -= t_vmin
            t_value /= t_vmax - t_vmin
            t_value = np.ma.masked_invalid(t_value, copy=False)
            return t_value[0] if is_scalar else t_value

        def inverse(self, value):
            if not self.scaled():
                raise ValueError('Not invertible until scaled')
            if self.vmin > self.vmax:
                raise ValueError('vmin must be less or equal to vmax')
            t_vmin, t_vmax = self._trf.transform([self.vmin, self.vmax])
            if not np.isfinite([t_vmin, t_vmax]).all():
                raise ValueError('Invalid vmin or vmax')
            value, is_scalar = self.process_value(value)
            rescaled = value * (t_vmax - t_vmin)
            rescaled += t_vmin
            value = self._trf.inverted().transform(rescaled).reshape(np.shape(value))
            return value[0] if is_scalar else value

        def autoscale_None(self, A):
            in_trf_domain = np.extract(np.isfinite(self._trf.transform(A)), A)
            if in_trf_domain.size == 0:
                in_trf_domain = np.ma.masked
            return super().autoscale_None(in_trf_domain)
    if base_norm_cls is Normalize:
        Norm.__name__ = f'{scale_cls.__name__}Norm'
        Norm.__qualname__ = f'{scale_cls.__qualname__}Norm'
    else:
        Norm.__name__ = base_norm_cls.__name__
        Norm.__qualname__ = base_norm_cls.__qualname__
    Norm.__module__ = base_norm_cls.__module__
    Norm.__doc__ = base_norm_cls.__doc__
    return Norm