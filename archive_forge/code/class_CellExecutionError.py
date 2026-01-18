from typing import Dict, List
from nbformat import NotebookNode
class CellExecutionError(CellControlSignal):
    """
    Custom exception to propagate exceptions that are raised during
    notebook execution to the caller. This is mostly useful when
    using nbconvert as a library, since it allows to deal with
    failures gracefully.
    """

    def __init__(self, traceback: str, ename: str, evalue: str) -> None:
        """Initialize the error."""
        super().__init__(traceback)
        self.traceback = traceback
        self.ename = ename
        self.evalue = evalue

    def __reduce__(self) -> tuple:
        """Reduce implementation."""
        return (type(self), (self.traceback, self.ename, self.evalue))

    def __str__(self) -> str:
        """Str repr."""
        if self.traceback:
            return self.traceback
        else:
            return f'{self.ename}: {self.evalue}'

    @classmethod
    def from_cell_and_msg(cls, cell: NotebookNode, msg: Dict) -> 'CellExecutionError':
        """Instantiate from a code cell object and a message contents
        (message is either execute_reply or error)
        """
        stream_outputs: List[str] = []
        for output in cell.outputs:
            if output['output_type'] == 'stream':
                stream_outputs.append(stream_output_msg.format(name=output['name'], text=output['text'].rstrip()))
        if stream_outputs:
            stream_outputs.insert(0, '')
            stream_outputs.append('------------------')
        stream_output: str = '\n'.join(stream_outputs)
        tb = '\n'.join(msg.get('traceback', []) or [])
        return cls(exec_err_msg.format(cell=cell, stream_output=stream_output, traceback=tb), ename=msg.get('ename', '<Error>'), evalue=msg.get('evalue', ''))