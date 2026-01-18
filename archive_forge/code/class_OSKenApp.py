import inspect
import itertools
import logging
import sys
import os
import gc
from os_ken import cfg
from os_ken import utils
from os_ken.controller.handler import register_instance, get_dependent_services
from os_ken.controller.controller import Datapath
from os_ken.controller import event
from os_ken.controller.event import EventRequestBase, EventReplyBase
from os_ken.lib import hub
from os_ken.ofproto import ofproto_protocol
class OSKenApp(object):
    """
    The base class for OSKen applications.

    OSKenApp subclasses are instantiated after osken-manager loaded
    all requested OSKen application modules.
    __init__ should call OSKenApp.__init__ with the same arguments.
    It's illegal to send any events in __init__.

    The instance attribute 'name' is the name of the class used for
    message routing among OSKen applications.  (Cf. send_event)
    It's set to __class__.__name__ by OSKenApp.__init__.
    It's discouraged for subclasses to override this.
    """
    _CONTEXTS = {}
    "\n    A dictionary to specify contexts which this OSKen application wants to use.\n    Its key is a name of context and its value is an ordinary class\n    which implements the context.  The class is instantiated by app_manager\n    and the instance is shared among OSKenApp subclasses which has _CONTEXTS\n    member with the same key.  A OSKenApp subclass can obtain a reference to\n    the instance via its __init__'s kwargs as the following.\n\n    Example::\n\n        _CONTEXTS = {\n            'network': network.Network\n        }\n\n        def __init__(self, *args, *kwargs):\n            self.network = kwargs['network']\n    "
    _EVENTS = []
    '\n    A list of event classes which this OSKenApp subclass would generate.\n    This should be specified if and only if event classes are defined in\n    a different python module from the OSKenApp subclass is.\n    '
    OFP_VERSIONS = None
    '\n    A list of supported OpenFlow versions for this OSKenApp.\n    The default is all versions supported by the framework.\n\n    Examples::\n\n        OFP_VERSIONS = [ofproto_v1_0.OFP_VERSION,\n                        ofproto_v1_2.OFP_VERSION]\n\n    If multiple OSKen applications are loaded in the system,\n    the intersection of their OFP_VERSIONS is used.\n    '

    @classmethod
    def context_iteritems(cls):
        """
        Return iterator over the (key, contxt class) of application context
        """
        return iter(cls._CONTEXTS.items())

    def __init__(self, *_args, **_kwargs):
        super(OSKenApp, self).__init__()
        self.name = self.__class__.__name__
        self.event_handlers = {}
        self.observers = {}
        self.threads = []
        self.main_thread = None
        self.events = hub.Queue(128)
        self._events_sem = hub.BoundedSemaphore(self.events.maxsize)
        if hasattr(self.__class__, 'LOGGER_NAME'):
            self.logger = logging.getLogger(self.__class__.LOGGER_NAME)
        else:
            self.logger = logging.getLogger(self.name)
        self.CONF = cfg.CONF

        class _EventThreadStop(event.EventBase):
            pass
        self._event_stop = _EventThreadStop()
        self.is_active = True

    def start(self):
        """
        Hook that is called after startup initialization is done.
        """
        self.threads.append(hub.spawn(self._event_loop))

    def stop(self):
        if self.main_thread:
            hub.kill(self.main_thread)
        self.is_active = False
        self._send_event(self._event_stop, None)
        hub.joinall(self.threads)

    def set_main_thread(self, thread):
        """
        Set self.main_thread so that stop() can terminate it.

        Only AppManager.instantiate_apps should call this function.
        """
        self.main_thread = thread

    def register_handler(self, ev_cls, handler):
        assert callable(handler)
        self.event_handlers.setdefault(ev_cls, [])
        self.event_handlers[ev_cls].append(handler)

    def unregister_handler(self, ev_cls, handler):
        assert callable(handler)
        self.event_handlers[ev_cls].remove(handler)
        if not self.event_handlers[ev_cls]:
            del self.event_handlers[ev_cls]

    def register_observer(self, ev_cls, name, states=None):
        states = states or set()
        ev_cls_observers = self.observers.setdefault(ev_cls, {})
        ev_cls_observers.setdefault(name, set()).update(states)

    def unregister_observer(self, ev_cls, name):
        observers = self.observers.get(ev_cls, {})
        observers.pop(name)

    def unregister_observer_all_event(self, name):
        for observers in self.observers.values():
            observers.pop(name, None)

    def observe_event(self, ev_cls, states=None):
        brick = _lookup_service_brick_by_ev_cls(ev_cls)
        if brick is not None:
            brick.register_observer(ev_cls, self.name, states)

    def unobserve_event(self, ev_cls):
        brick = _lookup_service_brick_by_ev_cls(ev_cls)
        if brick is not None:
            brick.unregister_observer(ev_cls, self.name)

    def get_handlers(self, ev, state=None):
        """Returns a list of handlers for the specific event.

        :param ev: The event to handle.
        :param state: The current state. ("dispatcher")
                      If None is given, returns all handlers for the event.
                      Otherwise, returns only handlers that are interested
                      in the specified state.
                      The default is None.
        """
        ev_cls = ev.__class__
        handlers = self.event_handlers.get(ev_cls, [])
        if state is None:
            return handlers

        def test(h):
            if not hasattr(h, 'callers') or ev_cls not in h.callers:
                return True
            states = h.callers[ev_cls].dispatchers
            if not states:
                return True
            return state in states
        return filter(test, handlers)

    def get_observers(self, ev, state):
        observers = []
        for k, v in self.observers.get(ev.__class__, {}).items():
            if not state or not v or state in v:
                observers.append(k)
        return observers

    def send_request(self, req):
        """
        Make a synchronous request.
        Set req.sync to True, send it to a OSKen application specified by
        req.dst, and block until receiving a reply.
        Returns the received reply.
        The argument should be an instance of EventRequestBase.
        """
        assert isinstance(req, EventRequestBase)
        req.sync = True
        req.reply_q = hub.Queue()
        self.send_event(req.dst, req)
        return req.reply_q.get()

    def _event_loop(self):
        while self.is_active or not self.events.empty():
            ev, state = self.events.get()
            self._events_sem.release()
            if ev == self._event_stop:
                continue
            handlers = self.get_handlers(ev, state)
            for handler in handlers:
                try:
                    handler(ev)
                except hub.TaskExit:
                    raise
                except:
                    LOG.exception('%s: Exception occurred during handler processing. Backtrace from offending handler [%s] servicing event [%s] follows.', self.name, handler.__name__, ev.__class__.__name__)

    def _send_event(self, ev, state):
        self._events_sem.acquire()
        self.events.put((ev, state))

    def send_event(self, name, ev, state=None):
        """
        Send the specified event to the OSKenApp instance specified by name.
        """
        if name in SERVICE_BRICKS:
            if isinstance(ev, EventRequestBase):
                ev.src = self.name
            LOG.debug('EVENT %s->%s %s', self.name, name, ev.__class__.__name__)
            SERVICE_BRICKS[name]._send_event(ev, state)
        else:
            LOG.debug('EVENT LOST %s->%s %s', self.name, name, ev.__class__.__name__)

    def send_event_to_observers(self, ev, state=None):
        """
        Send the specified event to all observers of this OSKenApp.
        """
        for observer in self.get_observers(ev, state):
            self.send_event(observer, ev, state)

    def reply_to_request(self, req, rep):
        """
        Send a reply for a synchronous request sent by send_request.
        The first argument should be an instance of EventRequestBase.
        The second argument should be an instance of EventReplyBase.
        """
        assert isinstance(req, EventRequestBase)
        assert isinstance(rep, EventReplyBase)
        rep.dst = req.src
        if req.sync:
            req.reply_q.put(rep)
        else:
            self.send_event(rep.dst, rep)

    def close(self):
        """
        teardown method.
        The method name, close, is chosen for python context manager
        """
        pass