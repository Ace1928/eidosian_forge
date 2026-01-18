def _install_external_importers():
    """Install importers that require external filesystem access"""
    global _bootstrap_external
    import _frozen_importlib_external
    _bootstrap_external = _frozen_importlib_external
    _frozen_importlib_external._install(sys.modules[__name__])