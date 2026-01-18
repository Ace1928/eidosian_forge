from bokeh.core.properties import Instance, List, Dict, String, Any
from bokeh.models import Tool, ColumnDataSource, PolyEditTool, PolyDrawTool
class PolyVertexDrawTool(PolyDrawTool):
    node_style = Dict(String, Any, help='\n    Custom styling to apply to the intermediate nodes of a patch or line glyph.')
    end_style = Dict(String, Any, help='\n    Custom styling to apply to the start and nodes of a patch or line glyph.')