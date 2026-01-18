def _warn_iro(self):
    if not self.WARN_BAD_IRO:
        return
    import warnings
    warnings.warn("An inconsistent resolution order is being requested. (Interfaces should follow the Python class rules known as C3.) For backwards compatibility, zope.interface will allow this, making the best guess it can to produce as meaningful an order as possible. In the future this might be an error. Set the warning filter to error, or set the environment variable 'ZOPE_INTERFACE_TRACK_BAD_IRO' to '1' and examine ro.C3.BAD_IROS to debug, or set 'ZOPE_INTERFACE_STRICT_IRO' to raise exceptions.", InconsistentResolutionOrderWarning)