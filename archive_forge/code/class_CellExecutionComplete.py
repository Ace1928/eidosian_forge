from typing import Dict, List
from nbformat import NotebookNode
class CellExecutionComplete(CellControlSignal):
    """
    Used as a control signal for cell execution across execute_cell and
    process_message function calls. Raised when all execution requests
    are completed and no further messages are expected from the kernel
    over zeromq channels.
    """
    pass