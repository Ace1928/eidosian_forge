from matplotlib import _api, backend_tools, cbook, widgets
class ToolEvent:
    """Event for tool manipulation (add/remove)."""

    def __init__(self, name, sender, tool, data=None):
        self.name = name
        self.sender = sender
        self.tool = tool
        self.data = data