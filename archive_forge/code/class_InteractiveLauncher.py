import inspect
from threading import Thread, Event
from kivy.app import App
from kivy.base import EventLoop
from kivy.clock import Clock
from kivy.utils import deprecated
class InteractiveLauncher(SafeMembrane):
    """
    Proxy to an application instance that launches it in a thread and
    then returns and acts as a proxy to the application in the thread.
    """
    __slots__ = ('_ref', 'safe', 'confirmed', 'thread', 'app')

    @deprecated
    def __init__(self, app=None, *args, **kwargs):
        if app is None:
            app = App()
        EventLoop.safe = Event()
        self.safe = EventLoop.safe
        self.safe.set()
        EventLoop.confirmed = Event()
        self.confirmed = EventLoop.confirmed
        self.app = app

        def startApp(app=app, *args, **kwargs):
            app.run(*args, **kwargs)
        self.thread = Thread(*args, target=startApp, **kwargs)

    def run(self):
        self.thread.start()
        self._ref = self.app

    def stop(self):
        EventLoop.quit = True
        self.thread.join()

    def __repr__(self):
        return self.app.__repr__()