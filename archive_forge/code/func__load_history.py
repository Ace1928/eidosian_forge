from IPython.core.history import HistoryAccessorBase
from traitlets import Dict, List
from queue import Empty  # Py 3
def _load_history(self, raw=True, output=False, hist_access_type='range', **kwargs):
    """
        Load the history over ZMQ from the kernel. Wraps the history
        messaging with loop to wait to get history results.
        """
    history = []
    if hasattr(self.client, 'history'):
        msg_id = self.client.history(raw=raw, output=output, hist_access_type=hist_access_type, **kwargs)
        while True:
            try:
                reply = self.client.get_shell_msg(timeout=1)
            except Empty:
                break
            else:
                if reply['parent_header'].get('msg_id') == msg_id:
                    history = reply['content'].get('history', [])
                    break
    return history