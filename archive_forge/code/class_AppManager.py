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
class AppManager(object):
    _instance = None

    @staticmethod
    def run_apps(app_lists):
        """Run a set of OSKen applications

        A convenient method to load and instantiate apps.
        This blocks until all relevant apps stop.
        """
        app_mgr = AppManager.get_instance()
        app_mgr.load_apps(app_lists)
        contexts = app_mgr.create_contexts()
        services = app_mgr.instantiate_apps(**contexts)
        try:
            hub.joinall(services)
        finally:
            app_mgr.close()
            for t in services:
                t.kill()
            hub.joinall(services)
            gc.collect()

    @staticmethod
    def get_instance():
        if not AppManager._instance:
            AppManager._instance = AppManager()
        return AppManager._instance

    def __init__(self):
        self.applications_cls = {}
        self.applications = {}
        self.contexts_cls = {}
        self.contexts = {}
        self.close_sem = hub.Semaphore()

    def load_app(self, name):
        mod = utils.import_module(name)
        clses = inspect.getmembers(mod, lambda cls: inspect.isclass(cls) and issubclass(cls, OSKenApp) and (mod.__name__ == cls.__module__))
        if clses:
            return clses[0][1]
        return None

    def load_apps(self, app_lists):
        app_lists = [app for app in itertools.chain.from_iterable((app.split(',') for app in app_lists))]
        while len(app_lists) > 0:
            app_cls_name = app_lists.pop(0)
            context_modules = [x.__module__ for x in self.contexts_cls.values()]
            if app_cls_name in context_modules:
                continue
            LOG.info('loading app %s', app_cls_name)
            cls = self.load_app(app_cls_name)
            if cls is None:
                continue
            self.applications_cls[app_cls_name] = cls
            services = []
            for key, context_cls in cls.context_iteritems():
                v = self.contexts_cls.setdefault(key, context_cls)
                assert v == context_cls
                context_modules.append(context_cls.__module__)
                if issubclass(context_cls, OSKenApp):
                    services.extend(get_dependent_services(context_cls))
            for i in get_dependent_services(cls):
                if i not in context_modules:
                    services.append(i)
            if services:
                app_lists.extend([s for s in set(services) if s not in app_lists])

    def create_contexts(self):
        for key, cls in self.contexts_cls.items():
            if issubclass(cls, OSKenApp):
                context = self._instantiate(None, cls)
            else:
                context = cls()
            LOG.info('creating context %s', key)
            assert key not in self.contexts
            self.contexts[key] = context
        return self.contexts

    def _update_bricks(self):
        for i in SERVICE_BRICKS.values():
            for _k, m in inspect.getmembers(i, inspect.ismethod):
                if not hasattr(m, 'callers'):
                    continue
                for ev_cls, c in m.callers.items():
                    if not c.ev_source:
                        continue
                    brick = _lookup_service_brick_by_mod_name(c.ev_source)
                    if brick:
                        brick.register_observer(ev_cls, i.name, c.dispatchers)
                    for brick in SERVICE_BRICKS.values():
                        if ev_cls in brick._EVENTS:
                            brick.register_observer(ev_cls, i.name, c.dispatchers)

    @staticmethod
    def _report_brick(name, app):
        LOG.debug('BRICK %s', name)
        for ev_cls, list_ in app.observers.items():
            LOG.debug('  PROVIDES %s TO %s', ev_cls.__name__, list_)
        for ev_cls in app.event_handlers.keys():
            LOG.debug('  CONSUMES %s', ev_cls.__name__)

    @staticmethod
    def report_bricks():
        for brick, i in SERVICE_BRICKS.items():
            AppManager._report_brick(brick, i)

    def _instantiate(self, app_name, cls, *args, **kwargs):
        LOG.info('instantiating app %s of %s', app_name, cls.__name__)
        if hasattr(cls, 'OFP_VERSIONS') and cls.OFP_VERSIONS is not None:
            ofproto_protocol.set_app_supported_versions(cls.OFP_VERSIONS)
        if app_name is not None:
            assert app_name not in self.applications
        app = cls(*args, **kwargs)
        register_app(app)
        assert app.name not in self.applications
        self.applications[app.name] = app
        return app

    def instantiate(self, cls, *args, **kwargs):
        app = self._instantiate(None, cls, *args, **kwargs)
        self._update_bricks()
        self._report_brick(app.name, app)
        return app

    def instantiate_apps(self, *args, **kwargs):
        for app_name, cls in self.applications_cls.items():
            self._instantiate(app_name, cls, *args, **kwargs)
        self._update_bricks()
        self.report_bricks()
        threads = []
        for app in self.applications.values():
            t = app.start()
            if t is not None:
                app.set_main_thread(t)
                threads.append(t)
        return threads

    @staticmethod
    def _close(app):
        close_method = getattr(app, 'close', None)
        if callable(close_method):
            close_method()

    def uninstantiate(self, name):
        app = self.applications.pop(name)
        unregister_app(app)
        for app_ in SERVICE_BRICKS.values():
            app_.unregister_observer_all_event(name)
        app.stop()
        self._close(app)
        events = app.events
        if not events.empty():
            app.logger.debug('%s events remains %d', app.name, events.qsize())

    def close(self):

        def close_all(close_dict):
            for app in close_dict.values():
                self._close(app)
            close_dict.clear()
        with self.close_sem:
            for app_name in list(self.applications.keys()):
                self.uninstantiate(app_name)
            assert not self.applications
            close_all(self.contexts)