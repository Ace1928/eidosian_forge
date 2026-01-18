from kivy.properties import AliasProperty, StringProperty, ColorProperty
from kivy.uix.behaviors import ToggleButtonBehavior
from kivy.uix.widget import Widget
class CheckBox(ToggleButtonBehavior, Widget):
    """CheckBox class, see module documentation for more information.
    """

    def _get_active(self):
        return self.state == 'down'

    def _set_active(self, value):
        self.state = 'down' if value else 'normal'
    active = AliasProperty(_get_active, _set_active, bind=('state',), cache=True)
    "Indicates if the switch is active or inactive.\n\n    :attr:`active` is a boolean and reflects and sets whether the underlying\n    :attr:`~kivy.uix.button.Button.state` is 'down' (True) or 'normal' (False).\n    It is a :class:`~kivy.properties.AliasProperty`, which accepts boolean\n    values and defaults to False.\n\n    .. versionchanged:: 1.11.0\n\n        It changed from a BooleanProperty to a AliasProperty.\n    "
    background_checkbox_normal = StringProperty('atlas://data/images/defaulttheme/checkbox_off')
    "Background image of the checkbox used for the default graphical\n    representation when the checkbox is not active.\n\n    .. versionadded:: 1.9.0\n\n    :attr:`background_checkbox_normal` is a\n    :class:`~kivy.properties.StringProperty` and defaults to\n    'atlas://data/images/defaulttheme/checkbox_off'.\n    "
    background_checkbox_down = StringProperty('atlas://data/images/defaulttheme/checkbox_on')
    "Background image of the checkbox used for the default graphical\n    representation when the checkbox is active.\n\n    .. versionadded:: 1.9.0\n\n    :attr:`background_checkbox_down` is a\n    :class:`~kivy.properties.StringProperty` and defaults to\n    'atlas://data/images/defaulttheme/checkbox_on'.\n    "
    background_checkbox_disabled_normal = StringProperty('atlas://data/images/defaulttheme/checkbox_disabled_off')
    "Background image of the checkbox used for the default graphical\n    representation when the checkbox is disabled and not active.\n\n    .. versionadded:: 1.9.0\n\n    :attr:`background_checkbox_disabled_normal` is a\n    :class:`~kivy.properties.StringProperty` and defaults to\n    'atlas://data/images/defaulttheme/checkbox_disabled_off'.\n    "
    background_checkbox_disabled_down = StringProperty('atlas://data/images/defaulttheme/checkbox_disabled_on')
    "Background image of the checkbox used for the default graphical\n    representation when the checkbox is disabled and active.\n\n    .. versionadded:: 1.9.0\n\n    :attr:`background_checkbox_disabled_down` is a\n    :class:`~kivy.properties.StringProperty` and defaults to\n    'atlas://data/images/defaulttheme/checkbox_disabled_on'.\n    "
    background_radio_normal = StringProperty('atlas://data/images/defaulttheme/checkbox_radio_off')
    "Background image of the radio button used for the default graphical\n    representation when the radio button is not active.\n\n    .. versionadded:: 1.9.0\n\n    :attr:`background_radio_normal` is a\n    :class:`~kivy.properties.StringProperty` and defaults to\n    'atlas://data/images/defaulttheme/checkbox_radio_off'.\n    "
    background_radio_down = StringProperty('atlas://data/images/defaulttheme/checkbox_radio_on')
    "Background image of the radio button used for the default graphical\n    representation when the radio button is active.\n\n    .. versionadded:: 1.9.0\n\n    :attr:`background_radio_down` is a\n    :class:`~kivy.properties.StringProperty` and defaults to\n    'atlas://data/images/defaulttheme/checkbox_radio_on'.\n    "
    background_radio_disabled_normal = StringProperty('atlas://data/images/defaulttheme/checkbox_radio_disabled_off')
    "Background image of the radio button used for the default graphical\n    representation when the radio button is disabled and not active.\n\n    .. versionadded:: 1.9.0\n\n    :attr:`background_radio_disabled_normal` is a\n    :class:`~kivy.properties.StringProperty` and defaults to\n    'atlas://data/images/defaulttheme/checkbox_radio_disabled_off'.\n    "
    background_radio_disabled_down = StringProperty('atlas://data/images/defaulttheme/checkbox_radio_disabled_on')
    "Background image of the radio button used for the default graphical\n    representation when the radio button is disabled and active.\n\n    .. versionadded:: 1.9.0\n\n    :attr:`background_radio_disabled_down` is a\n    :class:`~kivy.properties.StringProperty` and defaults to\n    'atlas://data/images/defaulttheme/checkbox_radio_disabled_on'.\n    "
    color = ColorProperty([1, 1, 1, 1])
    "Color is used for tinting the default graphical representation\n    of checkbox and radio button (images).\n\n    Color is in the format (r, g, b, a).\n\n    .. versionadded:: 1.10.0\n\n    :attr:`color` is a\n    :class:`~kivy.properties.ColorProperty` and defaults to\n    '[1, 1, 1, 1]'.\n\n    .. versionchanged:: 2.0.0\n        Changed from :class:`~kivy.properties.ListProperty` to\n        :class:`~kivy.properties.ColorProperty`.\n    "

    def __init__(self, **kwargs):
        self.fbind('state', self._on_state)
        super(CheckBox, self).__init__(**kwargs)

    def _on_state(self, instance, value):
        if self.group and self.state == 'down':
            self._release_group(self)

    def on_group(self, *largs):
        super(CheckBox, self).on_group(*largs)
        if self.active:
            self._release_group(self)