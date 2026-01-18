import sys
import threading
import wandb
def _posix_timed_input(prompt: str, timeout: float) -> str:
    _echo(prompt)
    sel = selectors.DefaultSelector()
    sel.register(sys.stdin, selectors.EVENT_READ, data=sys.stdin.readline)
    events = sel.select(timeout=timeout)
    for key, _ in events:
        input_callback = key.data
        input_data: str = input_callback()
        if not input_data:
            raise TimeoutError
        return input_data.rstrip(LF)
    _echo(LF)
    termios.tcflush(sys.stdin, termios.TCIFLUSH)
    raise TimeoutError