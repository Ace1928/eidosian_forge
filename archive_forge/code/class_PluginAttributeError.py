class PluginAttributeError(Exception):
    """A plugin threw an AttributeError while being lazily loaded."""
    pass