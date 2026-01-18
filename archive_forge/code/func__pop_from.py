import re
import threading
def _pop_from(self, lst, conf):
    popped = lst.pop()
    if conf is not None and popped is not conf:
        raise AssertionError(f'The config popped ({popped}) is not the same as the config expected ({conf})')