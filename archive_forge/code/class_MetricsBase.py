from os import environ
from kivy.utils import platform
from kivy.properties import AliasProperty
from kivy.event import EventDispatcher
from kivy.setupconfig import USE_SDL2
from kivy.context import register_context
from kivy._metrics import dpi2px, NUMERIC_FORMATS, dispatch_pixel_scale, \
class MetricsBase(EventDispatcher):
    """Class that contains the default attributes for Metrics. Don't use this
    class directly, but use the `Metrics` instance.
    """
    _dpi = _default_dpi
    _density = _default_density
    _fontscale = _default_fontscale

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fbind('dpi', dispatch_pixel_scale)
        self.fbind('density', dispatch_pixel_scale)
        self.fbind('fontscale', dispatch_pixel_scale)

    def get_dpi(self, force_recompute=False):
        if not force_recompute and self._dpi is not None:
            return self._dpi
        if platform == 'android':
            if USE_SDL2:
                import jnius
                Hardware = jnius.autoclass('org.renpy.android.Hardware')
                value = Hardware.getDPI()
            else:
                import android
                value = android.get_dpi()
        elif platform == 'ios':
            import ios
            value = ios.get_dpi()
        else:
            from kivy.base import EventLoop
            EventLoop.ensure_window()
            value = EventLoop.window.dpi
        sync_pixel_scale(dpi=value)
        return value

    def set_dpi(self, value):
        self._dpi = value
        sync_pixel_scale(dpi=value)
        return True
    dpi: float = AliasProperty(get_dpi, set_dpi, cache=True)
    'The DPI of the screen.\n\n    Depending on the platform, the DPI can be taken from the Window provider\n    (Desktop mainly) or from a platform-specific module (like android/ios).\n\n    :attr:`dpi` is a :class:`~kivy.properties.AliasProperty` and can be\n    set to change the value. But, the :attr:`density` is reloaded and reset if\n    we got it from the Window and the Window ``dpi`` changed.\n    '

    def get_dpi_rounded(self):
        dpi = self.dpi
        if dpi < 140:
            return 120
        elif dpi < 200:
            return 160
        elif dpi < 280:
            return 240
        return 320
    dpi_rounded: int = AliasProperty(get_dpi_rounded, None, bind=('dpi',), cache=True)
    'Return the :attr:`dpi` of the screen, rounded to the nearest of 120,\n    160, 240 or 320.\n\n    :attr:`dpi_rounded` is a :class:`~kivy.properties.AliasProperty` and\n    updates when :attr:`dpi` changes.\n    '

    def get_density(self, force_recompute=False):
        if not force_recompute and self._density is not None:
            return self._density
        value = 1.0
        if platform == 'android':
            import jnius
            Hardware = jnius.autoclass('org.renpy.android.Hardware')
            value = Hardware.metrics.scaledDensity
        elif platform == 'ios':
            import ios
            value = ios.get_scale()
        elif platform in ('macosx', 'win'):
            value = self.dpi / 96.0
        sync_pixel_scale(density=value)
        return value

    def set_density(self, value):
        self._density = value
        sync_pixel_scale(density=value)
        return True
    density: float = AliasProperty(get_density, set_density, bind=('dpi',), cache=True)
    'The density of the screen.\n\n    This value is 1 by default on desktops but varies on android depending on\n    the screen.\n\n    :attr:`density` is a :class:`~kivy.properties.AliasProperty` and can be\n    set to change the value. But, the :attr:`density` is reloaded and reset if\n    we got it from the Window and the Window ``density`` changed.\n    '

    def get_fontscale(self, force_recompute=False):
        if not force_recompute and self._fontscale is not None:
            return self._fontscale
        value = 1.0
        if platform == 'android':
            from jnius import autoclass
            if USE_SDL2:
                PythonActivity = autoclass('org.kivy.android.PythonActivity')
            else:
                PythonActivity = autoclass('org.renpy.android.PythonActivity')
            config = PythonActivity.mActivity.getResources().getConfiguration()
            value = config.fontScale
        sync_pixel_scale(fontscale=value)
        return value

    def set_fontscale(self, value):
        self._fontscale = value
        sync_pixel_scale(fontscale=value)
        return True
    fontscale: float = AliasProperty(get_fontscale, set_fontscale, cache=True)
    'The fontscale user preference.\n\n    This value is 1 by default but can vary between 0.8 and 1.2.\n\n    :attr:`fontscale` is a :class:`~kivy.properties.AliasProperty` and can be\n    set to change the value.\n    '

    def get_in(self):
        return dpi2px(1, 'in')
    inch: float = AliasProperty(get_in, None, bind=('dpi', 'density', 'fontscale'), cache=True)
    'The scaling factor that converts from inches to pixels.\n\n    :attr:`inch` is a :class:`~kivy.properties.AliasProperty` containing the\n    factor. E.g in KV: ``width: self.texture_size[0] + 10 * Metrics.inch`` will\n    update width when :attr:`inch` changes from a screen configuration change.\n    '

    def get_dp(self):
        return dpi2px(1, 'dp')
    dp: float = AliasProperty(get_dp, None, bind=('dpi', 'density', 'fontscale'), cache=True)
    'The scaling factor that converts from density-independent pixels to\n    pixels.\n\n    :attr:`dp` is a :class:`~kivy.properties.AliasProperty` containing the\n    factor. E.g in KV: ``width: self.texture_size[0] + 10 * Metrics.dp`` will\n    update width when :attr:`dp` changes from a screen configuration change.\n    '

    def get_sp(self):
        return dpi2px(1, 'sp')
    sp: float = AliasProperty(get_sp, None, bind=('dpi', 'density', 'fontscale'), cache=True)
    'The scaling factor that converts from scale-independent pixels to\n    pixels.\n\n    :attr:`sp` is a :class:`~kivy.properties.AliasProperty` containing the\n    factor. E.g in KV: ``width: self.texture_size[0] + 10 * Metrics.sp`` will\n    update width when :attr:`sp` changes from a screen configuration change.\n    '

    def get_pt(self):
        return dpi2px(1, 'pt')
    pt: float = AliasProperty(get_pt, None, bind=('dpi', 'density', 'fontscale'), cache=True)
    'The scaling factor that converts from points to pixels.\n\n    :attr:`pt` is a :class:`~kivy.properties.AliasProperty` containing the\n    factor. E.g in KV: ``width: self.texture_size[0] + 10 * Metrics.pt`` will\n    update width when :attr:`pt` changes from a screen configuration change.\n    '

    def get_cm(self):
        return dpi2px(1, 'cm')
    cm: float = AliasProperty(get_cm, None, bind=('dpi', 'density', 'fontscale'), cache=True)
    'The scaling factor that converts from centimeters to pixels.\n\n    :attr:`cm` is a :class:`~kivy.properties.AliasProperty` containing the\n    factor. E.g in KV: ``width: self.texture_size[0] + 10 * Metrics.cm`` will\n    update width when :attr:`cm` changes from a screen configuration change.\n    '

    def get_mm(self):
        return dpi2px(1, 'mm')
    mm: float = AliasProperty(get_mm, None, bind=('dpi', 'density', 'fontscale'), cache=True)
    'The scaling factor that converts from millimeters to pixels.\n\n    :attr:`mm` is a :class:`~kivy.properties.AliasProperty` containing the\n    factor. E.g in KV: ``width: self.texture_size[0] + 10 * Metrics.mm`` will\n    update width when :attr:`mm` changes from a screen configuration change.\n    '

    def reset_metrics(self):
        """Resets the dpi/density/fontscale to the platform values, overwriting
        any manually set values.
        """
        self.dpi = self.get_dpi(force_recompute=True)
        self.density = self.get_density(force_recompute=True)
        self.fontscale = self.get_fontscale(force_recompute=True)

    def reset_dpi(self, *args):
        """Resets the dpi (and possibly density) to the platform values,
        overwriting any manually set values.
        """
        self.dpi = self.get_dpi(force_recompute=True)

    def _set_cached_scaling(self):
        dispatch_pixel_scale()