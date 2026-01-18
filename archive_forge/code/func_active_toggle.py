from matplotlib import _api, backend_tools, cbook, widgets
@property
def active_toggle(self):
    """Currently toggled tools."""
    return self._toggled