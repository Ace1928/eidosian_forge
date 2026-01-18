from bokeh.core.properties import Instance, List, Dict, String, Any
from bokeh.models import Tool, ColumnDataSource, PolyEditTool, PolyDrawTool
class CheckpointTool(Tool):
    """
    Checkpoints the data on the supplied ColumnDataSources, allowing
    the RestoreTool to restore the data to a previous state.
    """
    sources = List(Instance(ColumnDataSource))