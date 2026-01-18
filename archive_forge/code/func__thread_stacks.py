import logging
import signal
import sys
import threading
import traceback
from pdb import Pdb
from scrapy.utils.engine import format_engine_status
from scrapy.utils.trackref import format_live_refs
def _thread_stacks(self):
    id2name = dict(((th.ident, th.name) for th in threading.enumerate()))
    dumps = ''
    for id_, frame in sys._current_frames().items():
        name = id2name.get(id_, '')
        dump = ''.join(traceback.format_stack(frame))
        dumps += f'# Thread: {name}({id_})\n{dump}\n'
    return dumps