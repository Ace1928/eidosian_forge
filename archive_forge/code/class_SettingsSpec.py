import sys
from collections import namedtuple
class SettingsSpec:
    """
    Runtime setting specification base class.

    SettingsSpec subclass objects used by `docutils.frontend.OptionParser`.
    """
    settings_spec = ()
    "Runtime settings specification.  Override in subclasses.\n\n    Defines runtime settings and associated command-line options, as used by\n    `docutils.frontend.OptionParser`.  This is a tuple of:\n\n    - Option group title (string or `None` which implies no group, just a list\n      of single options).\n\n    - Description (string or `None`).\n\n    - A sequence of option tuples.  Each consists of:\n\n      - Help text (string)\n\n      - List of option strings (e.g. ``['-Q', '--quux']``).\n\n      - Dictionary of keyword arguments sent to the OptionParser/OptionGroup\n        ``add_option`` method.\n\n        Runtime setting names are derived implicitly from long option names\n        ('--a-setting' becomes ``settings.a_setting``) or explicitly from the\n        'dest' keyword argument.\n\n        Most settings will also have a 'validator' keyword & function.  The\n        validator function validates setting values (from configuration files\n        and command-line option arguments) and converts them to appropriate\n        types.  For example, the ``docutils.frontend.validate_boolean``\n        function, **required by all boolean settings**, converts true values\n        ('1', 'on', 'yes', and 'true') to 1 and false values ('0', 'off',\n        'no', 'false', and '') to 0.  Validators need only be set once per\n        setting.  See the `docutils.frontend.validate_*` functions.\n\n        See the optparse docs for more details.\n\n    - More triples of group title, description, options, as many times as\n      needed.  Thus, `settings_spec` tuples can be simply concatenated.\n    "
    settings_defaults = None
    'A dictionary of defaults for settings not in `settings_spec` (internal\n    settings, intended to be inaccessible by command-line and config file).\n    Override in subclasses.'
    settings_default_overrides = None
    'A dictionary of auxiliary defaults, to override defaults for settings\n    defined in other components.  Override in subclasses.'
    relative_path_settings = ()
    'Settings containing filesystem paths.  Override in subclasses.\n    Settings listed here are to be interpreted relative to the current working\n    directory.'
    config_section = None
    'The name of the config file section specific to this component\n    (lowercase, no brackets).  Override in subclasses.'
    config_section_dependencies = None
    'A list of names of config file sections that are to be applied before\n    `config_section`, in order (from general to specific).  In other words,\n    the settings in `config_section` are to be overlaid on top of the settings\n    from these sections.  The "general" section is assumed implicitly.\n    Override in subclasses.'