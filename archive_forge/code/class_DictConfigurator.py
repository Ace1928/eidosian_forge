import errno
import io
import logging
import logging.handlers
import os
import queue
import re
import struct
import threading
import traceback
from socketserver import ThreadingTCPServer, StreamRequestHandler
class DictConfigurator(BaseConfigurator):
    """
    Configure logging using a dictionary-like object to describe the
    configuration.
    """

    def configure(self):
        """Do the configuration."""
        config = self.config
        if 'version' not in config:
            raise ValueError("dictionary doesn't specify a version")
        if config['version'] != 1:
            raise ValueError('Unsupported version: %s' % config['version'])
        incremental = config.pop('incremental', False)
        EMPTY_DICT = {}
        logging._acquireLock()
        try:
            if incremental:
                handlers = config.get('handlers', EMPTY_DICT)
                for name in handlers:
                    if name not in logging._handlers:
                        raise ValueError('No handler found with name %r' % name)
                    else:
                        try:
                            handler = logging._handlers[name]
                            handler_config = handlers[name]
                            level = handler_config.get('level', None)
                            if level:
                                handler.setLevel(logging._checkLevel(level))
                        except Exception as e:
                            raise ValueError('Unable to configure handler %r' % name) from e
                loggers = config.get('loggers', EMPTY_DICT)
                for name in loggers:
                    try:
                        self.configure_logger(name, loggers[name], True)
                    except Exception as e:
                        raise ValueError('Unable to configure logger %r' % name) from e
                root = config.get('root', None)
                if root:
                    try:
                        self.configure_root(root, True)
                    except Exception as e:
                        raise ValueError('Unable to configure root logger') from e
            else:
                disable_existing = config.pop('disable_existing_loggers', True)
                _clearExistingHandlers()
                formatters = config.get('formatters', EMPTY_DICT)
                for name in formatters:
                    try:
                        formatters[name] = self.configure_formatter(formatters[name])
                    except Exception as e:
                        raise ValueError('Unable to configure formatter %r' % name) from e
                filters = config.get('filters', EMPTY_DICT)
                for name in filters:
                    try:
                        filters[name] = self.configure_filter(filters[name])
                    except Exception as e:
                        raise ValueError('Unable to configure filter %r' % name) from e
                handlers = config.get('handlers', EMPTY_DICT)
                deferred = []
                for name in sorted(handlers):
                    try:
                        handler = self.configure_handler(handlers[name])
                        handler.name = name
                        handlers[name] = handler
                    except Exception as e:
                        if 'target not configured yet' in str(e.__cause__):
                            deferred.append(name)
                        else:
                            raise ValueError('Unable to configure handler %r' % name) from e
                for name in deferred:
                    try:
                        handler = self.configure_handler(handlers[name])
                        handler.name = name
                        handlers[name] = handler
                    except Exception as e:
                        raise ValueError('Unable to configure handler %r' % name) from e
                root = logging.root
                existing = list(root.manager.loggerDict.keys())
                existing.sort()
                child_loggers = []
                loggers = config.get('loggers', EMPTY_DICT)
                for name in loggers:
                    if name in existing:
                        i = existing.index(name) + 1
                        prefixed = name + '.'
                        pflen = len(prefixed)
                        num_existing = len(existing)
                        while i < num_existing:
                            if existing[i][:pflen] == prefixed:
                                child_loggers.append(existing[i])
                            i += 1
                        existing.remove(name)
                    try:
                        self.configure_logger(name, loggers[name])
                    except Exception as e:
                        raise ValueError('Unable to configure logger %r' % name) from e
                _handle_existing_loggers(existing, child_loggers, disable_existing)
                root = config.get('root', None)
                if root:
                    try:
                        self.configure_root(root)
                    except Exception as e:
                        raise ValueError('Unable to configure root logger') from e
        finally:
            logging._releaseLock()

    def configure_formatter(self, config):
        """Configure a formatter from a dictionary."""
        if '()' in config:
            factory = config['()']
            try:
                result = self.configure_custom(config)
            except TypeError as te:
                if "'format'" not in str(te):
                    raise
                config['fmt'] = config.pop('format')
                config['()'] = factory
                result = self.configure_custom(config)
        else:
            fmt = config.get('format', None)
            dfmt = config.get('datefmt', None)
            style = config.get('style', '%')
            cname = config.get('class', None)
            if not cname:
                c = logging.Formatter
            else:
                c = _resolve(cname)
            if 'validate' in config:
                result = c(fmt, dfmt, style, config['validate'])
            else:
                result = c(fmt, dfmt, style)
        return result

    def configure_filter(self, config):
        """Configure a filter from a dictionary."""
        if '()' in config:
            result = self.configure_custom(config)
        else:
            name = config.get('name', '')
            result = logging.Filter(name)
        return result

    def add_filters(self, filterer, filters):
        """Add filters to a filterer from a list of names."""
        for f in filters:
            try:
                if callable(f) or callable(getattr(f, 'filter', None)):
                    filter_ = f
                else:
                    filter_ = self.config['filters'][f]
                filterer.addFilter(filter_)
            except Exception as e:
                raise ValueError('Unable to add filter %r' % f) from e

    def configure_handler(self, config):
        """Configure a handler from a dictionary."""
        config_copy = dict(config)
        formatter = config.pop('formatter', None)
        if formatter:
            try:
                formatter = self.config['formatters'][formatter]
            except Exception as e:
                raise ValueError('Unable to set formatter %r' % formatter) from e
        level = config.pop('level', None)
        filters = config.pop('filters', None)
        if '()' in config:
            c = config.pop('()')
            if not callable(c):
                c = self.resolve(c)
            factory = c
        else:
            cname = config.pop('class')
            klass = self.resolve(cname)
            if issubclass(klass, logging.handlers.MemoryHandler) and 'target' in config:
                try:
                    th = self.config['handlers'][config['target']]
                    if not isinstance(th, logging.Handler):
                        config.update(config_copy)
                        raise TypeError('target not configured yet')
                    config['target'] = th
                except Exception as e:
                    raise ValueError('Unable to set target handler %r' % config['target']) from e
            elif issubclass(klass, logging.handlers.SMTPHandler) and 'mailhost' in config:
                config['mailhost'] = self.as_tuple(config['mailhost'])
            elif issubclass(klass, logging.handlers.SysLogHandler) and 'address' in config:
                config['address'] = self.as_tuple(config['address'])
            factory = klass
        kwargs = {k: config[k] for k in config if k != '.' and valid_ident(k)}
        try:
            result = factory(**kwargs)
        except TypeError as te:
            if "'stream'" not in str(te):
                raise
            kwargs['strm'] = kwargs.pop('stream')
            result = factory(**kwargs)
        if formatter:
            result.setFormatter(formatter)
        if level is not None:
            result.setLevel(logging._checkLevel(level))
        if filters:
            self.add_filters(result, filters)
        props = config.pop('.', None)
        if props:
            for name, value in props.items():
                setattr(result, name, value)
        return result

    def add_handlers(self, logger, handlers):
        """Add handlers to a logger from a list of names."""
        for h in handlers:
            try:
                logger.addHandler(self.config['handlers'][h])
            except Exception as e:
                raise ValueError('Unable to add handler %r' % h) from e

    def common_logger_config(self, logger, config, incremental=False):
        """
        Perform configuration which is common to root and non-root loggers.
        """
        level = config.get('level', None)
        if level is not None:
            logger.setLevel(logging._checkLevel(level))
        if not incremental:
            for h in logger.handlers[:]:
                logger.removeHandler(h)
            handlers = config.get('handlers', None)
            if handlers:
                self.add_handlers(logger, handlers)
            filters = config.get('filters', None)
            if filters:
                self.add_filters(logger, filters)

    def configure_logger(self, name, config, incremental=False):
        """Configure a non-root logger from a dictionary."""
        logger = logging.getLogger(name)
        self.common_logger_config(logger, config, incremental)
        logger.disabled = False
        propagate = config.get('propagate', None)
        if propagate is not None:
            logger.propagate = propagate

    def configure_root(self, config, incremental=False):
        """Configure a root logger from a dictionary."""
        root = logging.getLogger()
        self.common_logger_config(root, config, incremental)