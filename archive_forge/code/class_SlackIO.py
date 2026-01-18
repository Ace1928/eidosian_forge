import logging
from os import getenv
from ..auto import tqdm as tqdm_auto
from .utils_worker import MonoWorker
class SlackIO(MonoWorker):
    """Non-blocking file-like IO using a Slack app."""

    def __init__(self, token, channel):
        """Creates a new message in the given `channel`."""
        super(SlackIO, self).__init__()
        self.client = WebClient(token=token)
        self.text = self.__class__.__name__
        try:
            self.message = self.client.chat_postMessage(channel=channel, text=self.text)
        except Exception as e:
            tqdm_auto.write(str(e))
            self.message = None

    def write(self, s):
        """Replaces internal `message`'s text with `s`."""
        if not s:
            s = '...'
        s = s.replace('\r', '').strip()
        if s == self.text:
            return
        message = self.message
        if message is None:
            return
        self.text = s
        try:
            future = self.submit(self.client.chat_update, channel=message['channel'], ts=message['ts'], text='`' + s + '`')
        except Exception as e:
            tqdm_auto.write(str(e))
        else:
            return future