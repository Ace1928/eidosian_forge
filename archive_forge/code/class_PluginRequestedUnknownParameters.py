from __future__ import annotations
class PluginRequestedUnknownParameters(Flake8Exception):
    """The plugin requested unknown parameters."""
    FORMAT = '"%(name)s" requested unknown parameters causing %(exc)s'

    def __init__(self, plugin_name: str, exception: Exception) -> None:
        """Pop certain keyword arguments for initialization."""
        self.plugin_name = plugin_name
        self.original_exception = exception
        super().__init__(plugin_name, exception)

    def __str__(self) -> str:
        """Format our exception message."""
        return self.FORMAT % {'name': self.plugin_name, 'exc': self.original_exception}