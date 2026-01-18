from matplotlib import _api, backend_tools, cbook, widgets
class ToolTriggerEvent(ToolEvent):
    """Event to inform that a tool has been triggered."""

    def __init__(self, name, sender, tool, canvasevent=None, data=None):
        super().__init__(name, sender, tool, data)
        self.canvasevent = canvasevent