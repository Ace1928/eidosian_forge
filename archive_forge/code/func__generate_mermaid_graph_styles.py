import base64
import re
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple
from langchain_core.runnables.graph import (
def _generate_mermaid_graph_styles(node_colors: NodeColors) -> str:
    """Generates Mermaid graph styles for different node types."""
    styles = ''
    for class_name, color in asdict(node_colors).items():
        styles += f'\tclassDef {class_name}class fill:{color};\n'
    return styles