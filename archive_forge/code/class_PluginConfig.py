import importlib
from ..core.legacy_plugin_wrapper import LegacyPlugin
class PluginConfig:
    """Plugin Configuration Metadata

    This class holds the information needed to lazy-import plugins.

    Parameters
    ----------
    name : str
        The name of the plugin.
    class_name : str
        The name of the plugin class inside the plugin module.
    module_name : str
        The name of the module/package from which to import the plugin.
    is_legacy : bool
        If True, this plugin is a v2 plugin and will be wrapped in a
        LegacyPlugin. Default: False.
    package_name : str
        If the given module name points to a relative module, then the package
        name determines the package it is relative to.
    install_name : str
        The name of the optional dependency that can be used to install this
        plugin if it is missing.
    legacy_args : Dict
        A dictionary of kwargs to pass to the v2 plugin (Format) upon construction.

    Examples
    --------
    >>> PluginConfig(
            name="TIFF",
            class_name="TiffFormat",
            module_name="imageio.plugins.tifffile",
            is_legacy=True,
            install_name="tifffile",
            legacy_args={
                "description": "TIFF format",
                "extensions": ".tif .tiff .stk .lsm",
                "modes": "iIvV",
            },
        )
    >>> PluginConfig(
            name="pillow",
            class_name="PillowPlugin",
            module_name="imageio.plugins.pillow"
        )

    """

    def __init__(self, name, class_name, module_name, *, is_legacy=False, package_name=None, install_name=None, legacy_args=None):
        legacy_args = legacy_args or dict()
        self.name = name
        self.class_name = class_name
        self.module_name = module_name
        self.package_name = package_name
        self.is_legacy = is_legacy
        self.install_name = install_name or self.name
        self.legacy_args = {'name': name, 'description': 'A legacy plugin'}
        self.legacy_args.update(legacy_args)

    @property
    def format(self):
        """For backwards compatibility with FormatManager

        Delete when migrating to v3
        """
        if not self.is_legacy:
            raise RuntimeError('Can only get format for legacy plugins.')
        module = importlib.import_module(self.module_name, self.package_name)
        clazz = getattr(module, self.class_name)
        return clazz(**self.legacy_args)

    @property
    def plugin_class(self):
        """Get the plugin class (import if needed)

        Returns
        -------
        plugin_class : Any
            The class that can be used to instantiate plugins.

        """
        module = importlib.import_module(self.module_name, self.package_name)
        clazz = getattr(module, self.class_name)
        if self.is_legacy:
            legacy_plugin = clazz(**self.legacy_args)

            def partial_legacy_plugin(request):
                return LegacyPlugin(request, legacy_plugin)
            clazz = partial_legacy_plugin
        return clazz