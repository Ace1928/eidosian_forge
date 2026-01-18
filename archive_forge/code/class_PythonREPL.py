import functools
import logging
import multiprocessing
import sys
from io import StringIO
from typing import Dict, Optional
from langchain_core.pydantic_v1 import BaseModel, Field
class PythonREPL(BaseModel):
    """Simulates a standalone Python REPL."""
    globals: Optional[Dict] = Field(default_factory=dict, alias='_globals')
    locals: Optional[Dict] = Field(default_factory=dict, alias='_locals')

    @classmethod
    def worker(cls, command: str, globals: Optional[Dict], locals: Optional[Dict], queue: multiprocessing.Queue) -> None:
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        try:
            exec(command, globals, locals)
            sys.stdout = old_stdout
            queue.put(mystdout.getvalue())
        except Exception as e:
            sys.stdout = old_stdout
            queue.put(repr(e))

    def run(self, command: str, timeout: Optional[int]=None) -> str:
        """Run command with own globals/locals and returns anything printed.
        Timeout after the specified number of seconds."""
        warn_once()
        queue: multiprocessing.Queue = multiprocessing.Queue()
        if timeout is not None:
            p = multiprocessing.Process(target=self.worker, args=(command, self.globals, self.locals, queue))
            p.start()
            p.join(timeout)
            if p.is_alive():
                p.terminate()
                return 'Execution timed out'
        else:
            self.worker(command, self.globals, self.locals, queue)
        return queue.get()