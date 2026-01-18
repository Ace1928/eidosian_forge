import sys
import threading
import wandb
def _jupyter_timed_input(prompt: str, timeout: float) -> str:
    clear = True
    try:
        from IPython.core.display import clear_output
    except ImportError:
        clear = False
        wandb.termwarn("Unable to clear output, can't import clear_output from ipython.core")
    _echo(prompt)
    user_inp = None
    event = threading.Event()

    def get_input() -> None:
        nonlocal user_inp
        raw = input()
        if event.is_set():
            return
        user_inp = raw
    t = threading.Thread(target=get_input)
    t.start()
    t.join(timeout)
    event.set()
    if user_inp:
        return user_inp
    if clear:
        clear_output()
    raise TimeoutError